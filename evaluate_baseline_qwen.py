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
# 0. Argument Parsing (与 LLaDA 保持一致)
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Qwen/Llama Table Evaluation")
    
    parser.add_argument('--gpu_id', type=str, default='0', help='Logical GPU ID.')
    parser.add_argument('--dataset_path', type=str, default='data/table_llada_train_test.jsonl', help='Path to test dataset.')
    parser.add_argument('--log_dir', type=str, default='./logs/qwen_table_eval', help='Directory to save logs.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to Qwen/Llama model.')
    
    parser.add_argument("--shard_id", type=int, default=0, help="Current shard index")
    parser.add_argument("--num_shards", type=int, default=1, help="Total shards")
    parser.add_argument('--random_seed', type=int, default=42)
    
    parser.add_argument('--gen_length', type=int, default=512, help='Max new tokens.')
    
    return parser.parse_args()

# ==========================================
# 1. Logger Setup (与 LLaDA 保持一致)
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

# ==========================================
# 2. Metrics Utilities (与 LLaDA 保持一致)
# ==========================================
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def compute_f1(a_gold, a_pred):
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_metrics(gold, pred):
    em = 1 if normalize_answer(gold) == normalize_answer(pred) else 0
    f1 = compute_f1(gold, pred)
    return em, f1

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    args = parse_args()
    logger, case_file_path = setup_logging(args)
    random.seed(args.random_seed)
    
    logger.info(f"--- Init Baseline Eval {args.gpu_id} ---")
    
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
    
    logger.info(f"Total Test Samples: {total_samples}, Processing Chunk: [{start_idx}:{end_idx}]")

    # 2. Load Model (使用 AutoModelForCausalLM)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Baseline Model: {args.model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to(device).eval()
        
        if tokenizer.padding_side != 'left':
            tokenizer.padding_side = 'left'
             
    except Exception as e:
        logger.error(f"Model Load Failed: {e}")
        return

    # 3. Inference Loop
    total_em = 0
    total_f1 = 0
    processed_count = 0
    
    with open(case_file_path, 'w', encoding='utf-8') as f_case:
        for idx, item in enumerate(tqdm(my_data, desc=f"Eval-Baseline")):
            try:
                if 'messages' in item:
                    messages_input = [item['messages'][0]] 
                    ground_truth = item['messages'][1]['content']
                else: continue

                text_input = tokenizer.apply_chat_template(messages_input, add_generation_prompt=True, tokenize=False)
                encoded = tokenizer(text_input, return_tensors='pt', truncation=True, max_length=4000).to(device)
                
                # 标准自回归生成
                with torch.no_grad():
                    output_ids = model.generate(
                        **encoded,
                        max_new_tokens=args.gen_length,
                        do_sample=False,  # Greedy Search 为了公平对比
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # 只解码新生成的 token
                prediction = tokenizer.decode(output_ids[0][encoded.input_ids.shape[1]:], skip_special_tokens=True).strip()

                em, f1 = compute_metrics(ground_truth, prediction)
                total_em += em
                total_f1 += f1
                processed_count += 1

                logger.info(f"[{idx+1}] EM:{em} | F1:{f1:.2f} | Pred:{prediction[:50]}...")

                case_record = {
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "metrics": {"em": em, "f1": f1}
                }
                f_case.write(json.dumps(case_record) + "\n")
                f_case.flush()

            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                torch.cuda.empty_cache()
                continue

    # 4. Final Report
    if processed_count > 0:
        logger.info("\n" + "="*40)
        logger.info(f"Final Evaluation Report (Baseline)")
        logger.info("="*40)
        logger.info(f"Total Samples: {processed_count}")
        logger.info(f"Avg EM: {(total_em/processed_count)*100:.2f}%")
        logger.info(f"Avg F1: {(total_f1/processed_count)*100:.2f}%")
    
    logger.info("Evaluation Complete.")

if __name__ == "__main__":
    main()