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
    parser = argparse.ArgumentParser(description="Baseline Qwen/Llama WikiTQ Evaluation")
    
    parser.add_argument('--gpu_id', type=str, default='0', help='Logical GPU ID.')
    # 默认改为你的 WikiTQ 测试集路径
    parser.add_argument('--dataset_path', type=str, default='data/wikitq_test.jsonl', help='Path to test dataset.')
    parser.add_argument('--log_dir', type=str, default='./logs/wikitq_eval', help='Directory to save logs.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to Qwen/Llama model.')
    
    parser.add_argument("--shard_id", type=int, default=0, help="Current shard index")
    parser.add_argument("--num_shards", type=int, default=1, help="Total shards")
    parser.add_argument('--random_seed', type=int, default=42)
    
    # [关键修改] WikiTQ 答案很短，默认设为 64，强制模型短输出
    parser.add_argument('--gen_length', type=int, default=64, help='Max new tokens.')
    
    return parser.parse_args()

# ==========================================
# 1. Logger Setup
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
# 2. Metrics Utilities (字符串匹配评分机制)
# ==========================================
# 这部分就是你要的字符串匹配逻辑，非常适合 WikiTQ
def normalize_answer(s):
    """
    标准化：去除冠词、标点、统一大小写和空格
    例如：'The 2008.' -> '2008'
    """
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
    # EM: Exact Match (全匹配)
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
    
    logger.info(f"--- Init WikiTQ Eval {args.gpu_id} ---")
    
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

    # 2. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Model: {args.model_path}")
    
    try:
        # 既然你之前的代码能跑，就保持原来的加载方式
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
        for idx, item in enumerate(tqdm(my_data, desc=f"Eval-WikiTQ")):
            try:
                # [核心修改 1] 解析 WikiTQ 的 messages 结构
                messages = item.get('messages', [])
                
                # 查找 User 输入
                user_msg_obj = next((m for m in messages if m['role'] == 'user'), None)
                if not user_msg_obj: continue
                
                # 查找 Standard Answer (Assistant)
                assistant_msg_obj = next((m for m in messages if m['role'] == 'assistant'), None)
                if not assistant_msg_obj: continue 
                ground_truth = assistant_msg_obj['content']

                # [核心修改 2] Prompt 增强：强制要求短输出
                # 我们在原始表格/问题后面，追加一句指令
                raw_content = user_msg_obj['content']
                # 修改代码中的 instruction 变量
                instruction = (
                    "Read the table and answer the question. "
                    "Output ONLY the exact answer entity (e.g., a number, date, or name). "
                    "DO NOT output a full sentence. "
                    "DO NOT provide explanations or context. "
                    "Just the answer."
                )
                
                # 构造符合 Chat Template 的输入
                # 加入 System Prompt 进一步强化“只输出答案”的设定
                input_messages = [
                    {"role": "system", "content": "You are a helpful assistant. Answer the user's question concisely."},
                    {"role": "user", "content": raw_content + "\n\n" + instruction}
                ]

                # 应用 Chat Template
                text_input = tokenizer.apply_chat_template(input_messages, add_generation_prompt=True, tokenize=False)
                
                # Tokenize (WikiTQ 表格较长，这里给 4096 比较稳妥)
                encoded = tokenizer(text_input, return_tensors='pt', truncation=True, max_length=4096).to(device)
                
                # [核心修改 3] 生成参数控制
                with torch.no_grad():
                    output_ids = model.generate(
                        **encoded,
                        max_new_tokens=args.gen_length, # 默认为 64，物理限制它写作文
                        do_sample=False,  # Greedy Search
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # 解码
                input_len = encoded.input_ids.shape[1]
                prediction = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

                # 简单的后处理：只取第一行（防止模型输出答案后换行继续废话）
                prediction = prediction.split('\n')[0]

                # [评分机制] 使用代码自带的 EM/F1 进行字符串匹配
                em, f1 = compute_metrics(ground_truth, prediction)
                total_em += em
                total_f1 += f1
                processed_count += 1

                logger.info(f"[{idx+1}] EM:{em} | Pred: {prediction[:30]}... | Gold: {ground_truth[:30]}...")

                case_record = {
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "metrics": {"em": em, "f1": f1}
                }
                f_case.write(json.dumps(case_record, ensure_ascii=False) + "\n")
                f_case.flush()

            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                torch.cuda.empty_cache()
                continue

    # 4. Final Report
    if processed_count > 0:
        logger.info("\n" + "="*40)
        logger.info(f"Final Evaluation Report (WikiTQ)")
        logger.info("="*40)
        logger.info(f"Total Samples: {processed_count}")
        logger.info(f"Avg EM: {(total_em/processed_count)*100:.2f}%")
        logger.info(f"Avg F1: {(total_f1/processed_count)*100:.2f}%")
    
    logger.info("Evaluation Complete.")

if __name__ == "__main__":
    main()