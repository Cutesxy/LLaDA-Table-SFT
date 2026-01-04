import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM 
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
    parser = argparse.ArgumentParser(description="Baseline Llama WikiTQ Evaluation")
    
    parser.add_argument('--gpu_id', type=str, default='0', help='Logical GPU ID.')
    # Default to your WikiTQ test set
    parser.add_argument('--dataset_path', type=str, default='data/wikitq_test.jsonl', help='Path to test dataset.')
    parser.add_argument('--log_dir', type=str, default='./logs/llama_table_eval', help='Directory to save logs.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to Llama model.')
    
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=42)
    
    # [Alignment] Set to 64 to match Qwen and enforce short outputs for WikiTQ
    parser.add_argument('--gen_length', type=int, default=64, help='Max new tokens.')
    
    return parser.parse_args()

# ==========================================
# 1. Logger & Metrics (Identical to Qwen Script)
# ==========================================
def setup_logging(args):
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"eval_gpu{args.gpu_id}_metrics.log")
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
    case_file = os.path.join(args.log_dir, f"eval_gpu{args.gpu_id}_cases.jsonl")
    return logger, case_file

def normalize_answer(s):
    """Standard WikiTQ normalization"""
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
# 2. Main Execution
# ==========================================
def main():
    args = parse_args()
    logger, case_file_path = setup_logging(args)
    random.seed(args.random_seed)
    
    logger.info(f"--- Init Llama Baseline Eval {args.gpu_id} ---")
    
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
    
    logger.info(f"Total Samples: {total_samples}, Processing Chunk: [{start_idx}:{end_idx}]")

    # 2. Load Model
    # Llama usually requires device_map="auto" or explicit placement
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        # Llama specific: Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            torch_dtype=torch.bfloat16,
            device_map="auto" 
        ).eval()
        
        # Handle Llama-3 specific terminators
        terminators = [tokenizer.eos_token_id]
        if "<|eot_id|>" in tokenizer.all_special_tokens:
            terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
            
    except Exception as e:
        logger.error(f"Model Load Failed: {e}")
        return

    # 3. Inference Loop
    total_em, total_f1, processed_count = 0, 0, 0
    
    with open(case_file_path, 'w', encoding='utf-8') as f_case:
        for idx, item in enumerate(tqdm(my_data, desc=f"Eval-Llama")):
            try:
                # [Modification 1] Parse WikiTQ messages robustly
                messages = item.get('messages', [])
                
                # Find User Message
                user_msg_obj = next((m for m in messages if m['role'] == 'user'), None)
                if not user_msg_obj: continue
                
                # Find Ground Truth
                assistant_msg_obj = next((m for m in messages if m['role'] == 'assistant'), None)
                if not assistant_msg_obj: continue 
                ground_truth = assistant_msg_obj['content']

                # [Modification 2] Prompt Engineering (Same as Qwen)
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

                # Apply Template
                text_input = tokenizer.apply_chat_template(input_messages, add_generation_prompt=True, tokenize=False)
                
                # Tokenize
                model_inputs = tokenizer([text_input], return_tensors="pt", truncation=True, max_length=4096).to(model.device)

                # Generate
                with torch.no_grad():
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=args.gen_length, # Default 64
                        do_sample=False, 
                        eos_token_id=terminators,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                # Decode
                new_tokens = generated_ids[0][model_inputs.input_ids.shape[1]:]
                prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                
                # Post-process: Take first line
                prediction = prediction.split('\n')[0]

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
        logger.info(f"Final Report (Llama)")
        logger.info(f"Total: {processed_count}")
        logger.info(f"Avg EM: {(total_em/processed_count)*100:.2f}% | Avg F1: {(total_f1/processed_count)*100:.2f}%")

if __name__ == "__main__":
    main()