import os
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel 
from peft import PeftModel
import re
import argparse
import json
import random
import logging
import collections
import string
from tqdm import tqdm

# ==========================================
# 0. Argument Parsing
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="LLaDA-8B WikiTQ Evaluation (with LoRA support)")
    
    parser.add_argument('--gpu_id', type=str, default='0', help='Logical GPU ID.')
    # 默认路径改为 WikiTQ
    parser.add_argument('--dataset_path', type=str, default='data/wikitq_test.jsonl', help='Path to test dataset.')
    parser.add_argument('--log_dir', type=str, default='./logs/llada_table_eval', help='Directory to save logs.')
    
    # Base Model 路径
    parser.add_argument('--model_path', type=str, default="/home/zjusst/hxy/llada/models/GSAI-ML/LLaDA-8B-Instruct", help='Path to base model.')
    
    # LoRA Adapter 路径
    parser.add_argument('--adapter_path', type=str, default=None, help='Path to LoRA adapter checkpoint.')
    
    parser.add_argument("--shard_id", type=int, default=0, help="Current shard index")
    parser.add_argument("--num_shards", type=int, default=1, help="Total shards")
    parser.add_argument('--random_seed', type=int, default=42)
    
    # [关键修改] WikiTQ 答案很短，设为 64。
    # 对于 LLaDA (Diffusion)，减小这个值能显著提升速度。
    parser.add_argument('--gen_length', type=int, default=64, help='Generation length.')
    parser.add_argument('--steps', type=int, default=64, help='Diffusion steps (can be lower for short gen).')
    
    return parser.parse_args()

# ==========================================
# 1. Logger Setup
# ==========================================
def setup_logging(args):
    os.makedirs(args.log_dir, exist_ok=True)
    
    suffix = "_base"
    if args.adapter_path:
        ckpt_name = os.path.basename(os.path.normpath(args.adapter_path))
        suffix = f"_lora_{ckpt_name}"
        
    log_file = os.path.join(args.log_dir, f"eval_gpu{args.gpu_id}{suffix}_metrics.log")
    case_file = os.path.join(args.log_dir, f"eval_gpu{args.gpu_id}{suffix}_cases.jsonl")

    logger = logging.getLogger(f"Worker-{args.gpu_id}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger, case_file

# ==========================================
# 2. Metrics Utilities (Keep Identical)
# ==========================================
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def compute_f1(a_gold, a_pred):
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0: return int(gold_toks == pred_toks)
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)

def compute_metrics(gold, pred):
    em = 1 if normalize_answer(gold) == normalize_answer(pred) else 0
    f1 = compute_f1(gold, pred)
    return em, f1

# ==========================================
# 3. LLaDA Generation Logic (Standard)
# ==========================================
def add_gumbel_noise(logits, temperature):
    if temperature == 0: return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):
    
    # Construct input canvas
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    if gen_length < block_length: block_length = gen_length
    if gen_length % block_length != 0: block_length = gen_length 
    
    num_blocks = gen_length // block_length
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        for i in range(steps):
            mask_index = (x == mask_id)
            logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
    return x

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    args = parse_args()
    logger, case_file_path = setup_logging(args)
    random.seed(args.random_seed)
    
    logger.info(f"--- Init LLaDA Eval {args.gpu_id} ---")
    
    # 1. Load Data
    data = []
    try:
        with open(args.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Sharding
    random.shuffle(data)
    total_samples = len(data)
    chunk_size = total_samples // args.num_shards
    start_idx = args.shard_id * chunk_size
    end_idx = total_samples if args.shard_id == args.num_shards - 1 else start_idx + chunk_size
    my_data = data[start_idx:end_idx]
    
    logger.info(f"Chunk: [{start_idx}:{end_idx}] / {total_samples}")

    # 2. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Base: {args.model_path}")
    
    try:
        model = AutoModel.from_pretrained(
            args.model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.padding_side != 'left': tokenizer.padding_side = 'left'
        mask_id = 126336

        if args.adapter_path:
            logger.info(f"Loading LoRA: {args.adapter_path}")
            model = PeftModel.from_pretrained(model, args.adapter_path)
            # model.merge_and_unload() # Optional
        
        model.eval()
             
    except Exception as e:
        logger.error(f"Model Load Failed: {e}")
        return

    # 3. Inference Loop
    total_em, total_f1, processed_count = 0, 0, 0
    
    with open(case_file_path, 'w', encoding='utf-8') as f_case:
        
        for idx, item in enumerate(tqdm(my_data, desc=f"Eval-LLaDA")):
            try:
                # [核心修改 1] 鲁棒解析
                messages = item.get('messages', [])
                user_msg_obj = next((m for m in messages if m['role'] == 'user'), None)
                if not user_msg_obj: continue
                
                assistant_msg_obj = next((m for m in messages if m['role'] == 'assistant'), None)
                if not assistant_msg_obj: continue 
                ground_truth = assistant_msg_obj['content']

                # [核心修改 2] Prompt 对齐
                raw_content = user_msg_obj['content']
                # 修改代码中的 instruction 变量
                instruction = (
                    "Read the table and answer the question. "
                    "Output ONLY the exact answer entity (e.g., a number, date, or name). "
                    "DO NOT output a full sentence. "
                    "DO NOT provide explanations or context. "
                    "Just the answer."
                )
                
                input_messages = [
                    {"role": "system", "content": "You are a helpful assistant. Answer the user's question concisely."},
                    {"role": "user", "content": raw_content + "\n\n" + instruction}
                ]

                # Tokenize (WikiTQ 表格长，注意 max_length)
                text_input = tokenizer.apply_chat_template(input_messages, add_generation_prompt=True, tokenize=False)
                encoded = tokenizer(text_input, return_tensors='pt', truncation=True, max_length=4096).to(device)
                
                input_ids = encoded['input_ids']
                attention_mask = encoded['attention_mask']

                # [核心修改 3] 生成
                out_tokens = generate(
                    model=model, 
                    prompt=input_ids,
                    attention_mask=attention_mask,
                    steps=args.steps, 
                    gen_length=args.gen_length, # 默认为 64
                    block_length=args.gen_length, 
                    temperature=0.0, 
                    mask_id=mask_id
                )
                
                # 解码
                prediction = tokenizer.batch_decode(out_tokens[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                
                # 后处理
                prediction = prediction.strip().split('\n')[0]

                # Metrics
                em, f1 = compute_metrics(ground_truth, prediction)
                total_em += em
                total_f1 += f1
                processed_count += 1

                logger.info(f"[{idx+1}] EM:{em} | Pred: {prediction[:30]}... | Gold: {ground_truth[:30]}...")

                f_case.write(json.dumps({
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "metrics": {"em": em, "f1": f1}
                }, ensure_ascii=False) + "\n")
                f_case.flush()

            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                torch.cuda.empty_cache()
                continue

    if processed_count > 0:
        logger.info("\n" + "="*40)
        logger.info(f"Final Report (LLaDA)")
        logger.info(f"Mode: {'LoRA' if args.adapter_path else 'Base'}")
        logger.info(f"Total: {processed_count}")
        logger.info(f"Avg EM: {(total_em/processed_count)*100:.2f}% | Avg F1: {(total_f1/processed_count)*100:.2f}%")
    
    logger.info("Evaluation Complete.")

if __name__ == "__main__":
    main()