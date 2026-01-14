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
    parser = argparse.ArgumentParser(description="Unified Table Reasoning Evaluation (HF & LLaDA)")
    
    # ================= 核心配置 =================
    parser.add_argument('--model_type', type=str, required=True, choices=['hf', 'llada'], 
                        help="Model architecture type: 'hf' for Llama/Qwen, 'llada' for LLaDA")
    parser.add_argument('--model_path', type=str, required=True, help="Path to base model")
    parser.add_argument('--task', type=str, default='wtq', help="Task name (e.g., wtq, tabfact)")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to test dataset (.jsonl)")
    parser.add_argument('--log_dir', type=str, default='./logs/eval_result', help="Output directory")

    # ================= LLaDA 专属配置 =================
    parser.add_argument('--adapter_path', type=str, default=None, help="[LLaDA Only] Path to LoRA adapter")
    parser.add_argument('--steps', type=int, default=64, help="[LLaDA Only] Diffusion steps")
    
    # ================= 并行与生成配置 =================
    parser.add_argument('--gpu_id', type=str, default='0', help="Logical GPU ID for logging")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard index for data parallelism")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument('--gen_length', type=int, default=64, help="Max new tokens to generate")
    parser.add_argument('--random_seed', type=int, default=42)

    return parser.parse_args()

def setup_logging(args):
    os.makedirs(args.log_dir, exist_ok=True)
    
    suffix = ""
    if args.adapter_path:
        ckpt_name = os.path.basename(os.path.normpath(args.adapter_path))
        suffix = f"_lora_{ckpt_name}"
    
    log_filename = f"eval_gpu{args.gpu_id}_{args.model_type}_{args.task}{suffix}"
    
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
    logger.info(f"   Model Type: {args.model_type.upper()}")
    logger.info(f"   GPU ID: {args.gpu_id}")
    logger.info("="*40)

    # -------------------------------------------------------
    # 1. 加载任务
    # -------------------------------------------------------
    try:
        task = get_task(args.task, args.dataset_path)
        logger.info(f"Task '{args.task}' loaded. Total data: {len(task.data)}")
    except Exception as e:
        logger.error(f"Failed to load task: {e}")
        return

    # -------------------------------------------------------
    # 2. 数据分片
    # -------------------------------------------------------
    total_samples = len(task.data)
    chunk_size = total_samples // args.num_shards
    start_idx = args.shard_id * chunk_size
    end_idx = total_samples if args.shard_id == args.num_shards - 1 else start_idx + chunk_size
    
    my_data = task.data[start_idx:end_idx]
    logger.info(f"Processing Shard {args.shard_id}/{args.num_shards}: indices [{start_idx}:{end_idx}] ({len(my_data)} samples)")

    # -------------------------------------------------------
    # 3. 加载模型
    # -------------------------------------------------------
    try:
        if args.model_type == 'llada':
            model = LLaDAModelWrapper(
                model_path=args.model_path,
                adapter_path=args.adapter_path,
                steps=args.steps,
                gen_length=args.gen_length
            )
        elif args.model_type == 'hf':
            model = HFModelWrapper(
                model_path=args.model_path
            )
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # -------------------------------------------------------
    # 4. 评测循环
    # -------------------------------------------------------
    # 初始化统计变量 (兼容不同任务)
    total_em = 0.0
    total_f1 = 0.0
    total_acc = 0.0
    processed_count = 0
    
    logger.info("Starting inference...")
    
    with open(case_file_path, 'w', encoding='utf-8') as f_case:
        for idx, sample in enumerate(tqdm(my_data, desc=f"Eval GPU{args.gpu_id}")):
            try:
                # A. 准备输入
                messages = task.build_messages(sample)
                ground_truth = task.get_ground_truth(sample)
                
                if not messages or not ground_truth:
                    continue
                
                # B. 模型生成
                raw_output = model.generate(messages, max_new_tokens=args.gen_length)
                
                # C. 后处理 (Task Specific)
                prediction = task.post_process(raw_output)
                
                # D. 计算指标
                metrics = task.compute_metrics(ground_truth, prediction)
                
                # E. 统计积累 (使用 .get 安全获取)
                total_em += metrics.get('em', 0)
                total_f1 += metrics.get('f1', 0)
                total_acc += metrics.get('accuracy', 0)
                
                processed_count += 1
                
                # 构建日志字符串
                log_msg = f"[{idx}] "
                if 'em' in metrics: log_msg += f"EM:{metrics['em']} "
                if 'accuracy' in metrics: log_msg += f"Acc:{metrics['accuracy']} "
                log_msg += f"| Pred: {prediction} | Gold: {ground_truth}"
                
                # 打印日志 (防止日志过大，这里只打印，如果需要可以每10条打印一次)
                logger.info(log_msg)
                
                # 写入结果文件
                record = {
                    "id": idx + start_idx,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "raw_output": raw_output, # 保存原始输出方便 debug
                    "metrics": metrics
                }
                f_case.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_case.flush()

            except Exception as e:
                logger.error(f"Error processing sample index {idx}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
    
    # -------------------------------------------------------
    # 5. 最终报告
    # -------------------------------------------------------
    if processed_count > 0:
        logger.info("\n" + "="*40)
        logger.info(f"Shard {args.shard_id} Completed.")
        logger.info(f"Total Processed: {processed_count}")
        
        if total_em > 0 or 'wtq' in args.task.lower():
            avg_em = (total_em / processed_count) * 100
            avg_f1 = (total_f1 / processed_count) * 100
            logger.info(f"Average EM: {avg_em:.2f}%")
            logger.info(f"Average F1: {avg_f1:.2f}%")
        
        if total_acc > 0 or 'tabfact' in args.task.lower():
            avg_acc = (total_acc / processed_count) * 100
            logger.info(f"Average Accuracy: {avg_acc:.2f}%")
        
        logger.info("="*40)
    else:
        logger.warning("No samples were processed successfully.")

if __name__ == "__main__":
    main()