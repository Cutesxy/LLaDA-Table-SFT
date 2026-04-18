import os
import argparse
import sys

# ==========================================
# [关键修改 1] 在 import vllm 之前就锁定显卡
# 防止 vLLM 提前扫描导致死锁或抢卡
# ==========================================
def parse_args_peek():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0')
    # 只解析 gpu_id，忽略其他参数
    args, _ = parser.parse_known_args()
    return args

# 提前设置环境变量
temp_args = parse_args_peek()
os.environ["CUDA_VISIBLE_DEVICES"] = temp_args.gpu_id
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# ==========================================
# 现在再 import 库
# ==========================================
import json
import random
import logging
import re
import string
import collections
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ... (parse_args, setup_logging 等函数保持不变，或者直接用下面的完整版) ...
def parse_args():
    parser = argparse.ArgumentParser(description="Llama-3.1-8B Baseline Eval (vLLM)")
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default='./logs/llama_table_eval')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--gen_length', type=int, default=512)
    return parser.parse_args()

def setup_logging(args):
    # 这里不需要再设置 os.environ 了，前面已经设过了
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"eval_gpu{args.gpu_id}_metrics.log")
    logger = logging.getLogger(f"Worker-{args.gpu_id}")
    logger.setLevel(logging.INFO)
    logger.handlers = [] # 清空句柄
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    case_file = os.path.join(args.log_dir, f"eval_gpu{args.gpu_id}_cases.jsonl")
    return logger, case_file

# ... (compute_metrics 等工具函数保持不变) ...
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def compute_metrics(gold, pred):
    gold_toks = normalize_answer(gold).split()
    pred_toks = normalize_answer(pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    em = int(normalize_answer(gold) == normalize_answer(pred))
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        f1 = int(gold_toks == pred_toks)
    elif num_same == 0:
        f1 = 0
    else:
        precision = num_same / len(pred_toks)
        recall = num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
    return em, f1

def main():
    args = parse_args()
    logger, case_file_path = setup_logging(args)
    random.seed(args.random_seed)

    logger.info(f"--- Init vLLM Eval on Physical GPU {args.gpu_id} ---")

    # Load Data
    data = []
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    
    random.shuffle(data)
    total_samples = len(data)
    chunk_size = total_samples // args.num_shards
    start_idx = args.shard_id * chunk_size
    end_idx = total_samples if args.shard_id == args.num_shards - 1 else start_idx + chunk_size
    my_data = data[start_idx:end_idx]
    
    logger.info(f"Processing Shard {args.shard_id}/{args.num_shards}: {len(my_data)} samples")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # [关键修改 2] 限制上下文长度 & 调整精度
    logger.info("Loading vLLM engine...")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=1, 
        
        # [修改点] 强制使用 float16，为了兼容 V100。
        # 如果全是 4090，这里可以用 "bfloat16"，但 "float16" 是最通用的
        dtype="auto", 
        
        # [修改点] 必须限制 Llama 3.1 的长度，否则 24G 显卡扛不住 128k 上下文的初始化
        max_model_len=8192, 
        
        # [修改点] 稍微降低一点显存占用，给 I/O 留点空间
        gpu_memory_utilization=0.9, 
        
        enforce_eager=False
    )
    
    stop_token_ids = [tokenizer.eos_token_id]
    if "<|eot_id|>" in tokenizer.all_special_tokens:
        stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.gen_length,
        stop_token_ids=stop_token_ids
    )

    logger.info("Building prompts...")
    prompts = []
    valid_indices = []
    
    for i, item in enumerate(my_data):
        if "messages" in item:
            prompt = tokenizer.apply_chat_template(
                [item["messages"][0]], 
                add_generation_prompt=True, 
                tokenize=False
            )
            prompts.append(prompt)
            valid_indices.append(i)

    logger.info(f"Start Batch Generation for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    total_em, total_f1 = 0, 0
    with open(case_file_path, "w", encoding="utf-8") as f_case:
        for i, output in enumerate(outputs):
            original_idx = valid_indices[i]
            ground_truth = my_data[original_idx]["messages"][1]["content"]
            prediction = output.outputs[0].text.strip()
            
            em, f1 = compute_metrics(ground_truth, prediction)
            total_em += em
            total_f1 += f1
            
            f_case.write(json.dumps({
                "ground_truth": ground_truth,
                "prediction": prediction,
                "metrics": {"em": em, "f1": f1}
            }, ensure_ascii=False) + "\n")
            
            if i < 3:
                logger.info(f"Sample [{i}] EM:{em} | Pred: {prediction[:50]}...")

    if len(outputs) > 0:
        logger.info(f"Final EM: {100 * total_em / len(outputs):.2f}%")
        logger.info(f"Final F1: {100 * total_f1 / len(outputs):.2f}%")

if __name__ == "__main__":
    main()