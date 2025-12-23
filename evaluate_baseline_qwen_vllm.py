import os
import re
import json
import random
import argparse
import logging
import collections
import string
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ==========================================
# 0. Argument Parsing
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Qwen/Llama Table Evaluation (vLLM)")

    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--dataset_path', type=str, default='data/table_llada_train_test.jsonl')
    parser.add_argument('--log_dir', type=str, default='./logs/qwen_table_eval')
    parser.add_argument('--model_path', type=str, required=True)

    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=42)

    parser.add_argument('--gen_length', type=int, default=512)

    return parser.parse_args()

# ==========================================
# 1. Logger
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
# 2. Metrics
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
    if num_same == 0:
        return 0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)

def compute_metrics(gold, pred):
    em = int(normalize_answer(gold) == normalize_answer(pred))
    f1 = compute_f1(gold, pred)
    return em, f1

# ==========================================
# 3. Main
# ==========================================
def main():
    args = parse_args()
    logger, case_file_path = setup_logging(args)
    random.seed(args.random_seed)

    logger.info(f"--- Init Baseline Eval (vLLM) GPU {args.gpu_id} ---")

    # -------- Load Data --------
    data = []
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    random.shuffle(data)
    total_samples = len(data)

    chunk_size = total_samples // args.num_shards
    start_idx = args.shard_id * chunk_size
    end_idx = total_samples if args.shard_id == args.num_shards - 1 else start_idx + chunk_size
    my_data = data[start_idx:end_idx]

    logger.info(f"Total Samples: {total_samples}, Shard [{start_idx}:{end_idx}]")

    # -------- Tokenizer --------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    # -------- vLLM Model --------
    logger.info(f"Loading model with vLLM: {args.model_path}")

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.gen_length,
    )

    # -------- Build Prompts --------
    prompts = []
    gold_answers = []

    for item in my_data:
        if 'messages' not in item:
            continue
        messages_input = [item['messages'][0]]
        gold_answers.append(item['messages'][1]['content'])

        prompt = tokenizer.apply_chat_template(
            messages_input,
            add_generation_prompt=True,
            tokenize=False
        )
        prompts.append(prompt)

    logger.info(f"Prepared {len(prompts)} prompts, start vLLM generation...")

    # -------- Inference (Batched & Fast) --------
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

            logger.info(f"[{i+1}] EM:{em} | F1:{f1:.2f} | Pred:{pred[:50]}...")

            f_case.write(json.dumps({
                "ground_truth": gold,
                "prediction": pred,
                "metrics": {"em": em, "f1": f1}
            }) + "\n")

    # -------- Final Report --------
    logger.info("\n" + "=" * 40)
    logger.info("Final Evaluation Report (vLLM)")
    logger.info("=" * 40)
    logger.info(f"Total Samples: {processed_count}")
    logger.info(f"Avg EM: {100 * total_em / processed_count:.2f}%")
    logger.info(f"Avg F1: {100 * total_f1 / processed_count:.2f}%")
    logger.info("Evaluation Complete.")

if __name__ == "__main__":
    main()
