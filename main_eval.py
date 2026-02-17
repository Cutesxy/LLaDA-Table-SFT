import argparse
import os
import json
import logging
import random
import torch
from tqdm import tqdm

# 引入模块
from eval_core.tasks import get_task
from eval_core.models import HFModelWrapper, LLaDAModelWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Table Reasoning Evaluation")
    
    # ... (保持不变) ...
    parser.add_argument('--model_type', type=str, required=True, choices=['hf', 'llada'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--task', type=str, default='tabfact-completion')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default='./logs/eval_result')

    # LLaDA 专属参数
    parser.add_argument('--adapter_path', type=str, default=None)
    parser.add_argument('--steps', type=int, default=64)
    
    # [核心修改 1] 增加 block_size 参数
    # 默认值设为 32，这是 Block Diffusion 论文和您训练时的标准设置
    parser.add_argument('--block_size', type=int, default=32, help="Block size for diffusion generation (Must match training!)")
    
    # 并行配置
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    
    parser.add_argument('--gen_length', type=int, default=32, help="Total generation length")
    
    parser.add_argument('--random_seed', type=int, default=42)

    return parser.parse_args()

def setup_logging(args):
    os.makedirs(args.log_dir, exist_ok=True)
    suffix = f"_lora_{os.path.basename(os.path.normpath(args.adapter_path))}" if args.adapter_path else ""
    # 日志文件名带上 block_size 方便区分
    log_filename = f"eval_gpu{args.gpu_id}_{args.model_type}_{args.task}_len{args.gen_length}_blk{args.block_size}{suffix}"
    
    log_file = os.path.join(args.log_dir, f"{log_filename}.log")
    case_file = os.path.join(args.log_dir, f"{log_filename}_cases.jsonl")

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

def main():
    args = parse_args()
    logger, case_file_path = setup_logging(args)
    
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    logger.info("="*40)
    logger.info(f"   Task: {args.task.upper()}")
    logger.info(f"   Model: {args.model_type.upper()}")
    logger.info(f"   Gen Length: {args.gen_length}")
    logger.info(f"   Block Size: {args.block_size}") # 打印 Block Size
    logger.info("="*40)

    # 1. 加载任务
    try:
        task = get_task(args.task, args.dataset_path)
    except Exception as e:
        logger.error(f"Failed to load task: {e}")
        return

    # 2. 数据分片
    total_samples = len(task.data)
    chunk_size = total_samples // args.num_shards
    start_idx = args.shard_id * chunk_size
    end_idx = total_samples if args.shard_id == args.num_shards - 1 else start_idx + chunk_size
    my_data = task.data[start_idx:end_idx]
    logger.info(f"Processing Shard {args.shard_id}/{args.num_shards} ({len(my_data)} samples)")

    # 3. 加载模型
    try:
        if args.model_type == 'llada':
            model = LLaDAModelWrapper(
                model_path=args.model_path,
                adapter_path=args.adapter_path,
                steps=args.steps,
                gen_length=args.gen_length,
                # [核心修改 2] 将 block_size 传入模型初始化
                block_length=args.block_size 
            )
        elif args.model_type == 'hf':
            model = HFModelWrapper(model_path=args.model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 4. 评测循环
    total_acc = 0.0
    processed_count = 0
    current_gen_len = args.gen_length
    
    logger.info(f"Starting inference with Gen Length = {current_gen_len}, Block Size = {args.block_size}...")
    
    with open(case_file_path, 'w', encoding='utf-8') as f_case:
        for idx, sample in enumerate(tqdm(my_data, desc=f"Eval GPU{args.gpu_id}")):
            try:
                # A. 准备输入
                messages = task.build_messages(sample)
                ground_truth = task.get_ground_truth(sample)
                if not messages: continue
                
                # B. 模型生成
                # 返回 Token IDs Tensor [1, seq_len]
                raw_output_ids = model.generate(messages, max_new_tokens=current_gen_len)
                
                # C. EOS 截断逻辑
                output_seq = raw_output_ids[0]
                
                if model.tokenizer.eos_token_id in output_seq:
                    eos_idx = (output_seq == model.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
                    valid_ids = output_seq[:eos_idx]
                else:
                    valid_ids = output_seq
                
                # 解码
                prediction = model.tokenizer.decode(valid_ids, skip_special_tokens=True).strip()
                
                # D. 计算指标
                metrics = task.compute_metrics(ground_truth, prediction)
                
                total_acc += metrics.get('accuracy', 0)
                processed_count += 1
                
                log_msg = f"[{idx}] "
                if 'accuracy' in metrics: log_msg += f"Acc:{metrics['accuracy']} "
                log_msg += f"| Pred: {prediction} | GT: {ground_truth}"
                logger.info(log_msg)
                
                record = {
                    "id": idx + start_idx,
                    "ground_truth": ground_truth,
                    "mask_len": current_gen_len,
                    "prediction": prediction,
                    "metrics": metrics
                }
                f_case.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_case.flush()

            except Exception as e:
                logger.error(f"Error processing sample index {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if processed_count > 0:
        avg_acc = (total_acc / processed_count) * 100
        logger.info(f"Shard {args.shard_id} Finished. Avg Accuracy: {avg_acc:.2f}%")

if __name__ == "__main__":
    main()