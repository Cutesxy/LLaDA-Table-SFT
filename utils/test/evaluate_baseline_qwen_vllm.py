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

# ... (parse_args, setup_logging 等函数保持不变) ...
def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Qwen Table Evaluation (vLLM)")
    parser.add_argument('--gpu_id', type=str, default='0', help="Physical GPU ID")
    parser.add_argument('--dataset_path', type=str, default='data/table_llada_train_test.jsonl')
    parser.add_argument('--log_dir', type=str, default='./logs/qwen_table_eval')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--gen_length', type=int, default=512)
    return parser.parse_args()

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

# ... (Metrics 工具函数保持不变) ...
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

    logger.info(f"--- Init Baseline Eval (vLLM) on Physical GPU {args.gpu_id} ---")

    # Load Data
    data = []
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))

    # Sharding
    random.shuffle(data)
    total_samples = len(data)
    chunk_size = total_samples // args.num_shards
    start_idx = args.shard_id * chunk_size
    end_idx = total_samples if args.shard_id == args.num_shards - 1 else start_idx + chunk_size
    my_data = data[start_idx:end_idx]

    logger.info(f"Total Samples: {total_samples}, Processing Shard [{start_idx}:{end_idx}]")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    logger.info(f"Loading model with vLLM: {args.model_path}")
    
    # [关键修改 2] 适配 V100 和 防止 OOM
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=1, 
        
        # [修改点] 强制使用 float16，否则 V100 会报错
        dtype="auto", 
        
        # [修改点] 强制限制上下文长度，否则 Qwen 32k/128k 会撑爆 16GB 显存
        max_model_len=8192, 
        
        # [修改点] V100 16GB 建议 0.9，如果还崩就改 0.85
        gpu_memory_utilization=0.9, 
        
        enforce_eager=False
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.gen_length,
        # Qwen 的 EOS ID 通常不需要额外指定，vLLM 会自动读 config
        # 但显式写上也无妨
        stop_token_ids=[tokenizer.eos_token_id] 
    )

    logger.info("Building prompts...")
    prompts = []
    gold_answers = []

    for item in my_data:
        if 'messages' not in item: continue
        messages_input = [item['messages'][0]]
        gold_answers.append(item['messages'][1]['content'])

        prompt = tokenizer.apply_chat_template(
            messages_input, add_generation_prompt=True, tokenize=False
        )
        prompts.append(prompt)

    logger.info(f"Start vLLM generation for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    total_em, total_f1 = 0, 0
    processed_count = 0

    with open(case_file_path, 'w', encoding='utf-8') as f_case:
        for i, output in enumerate(outputs):
            pred = output.outputs[0].text.strip()
            gold = gold_answers[i]

            em, f1 = compute_metrics(gold, pred)
            total_em += em
            total_f1 += f1
            processed_count += 1

            if i < 3:
                logger.info(f"Sample [{i}] EM:{em} | Pred:{pred[:50]}...")

            f_case.write(json.dumps({
                "ground_truth": gold,
                "prediction": pred,
                "metrics": {"em": em, "f1": f1}
            }, ensure_ascii=False) + "\n")

    if processed_count > 0:
        logger.info("\n" + "=" * 40)
        logger.info(f"Final EM: {100 * total_em / processed_count:.2f}%")
        logger.info(f"Final F1: {100 * total_f1 / processed_count:.2f}%")
        logger.info("=" * 40)

if __name__ == "__main__":
    main()