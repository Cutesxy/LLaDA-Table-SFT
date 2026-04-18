import re
import sys
import os

def calculate_scores(log_file_path):
    if not os.path.exists(log_file_path):
        print(f"错误: 找不到文件 {log_file_path}")
        return

    em_total = 0.0
    f1_total = 0.0
    count = 0

    # 匹配模式：寻找类似 EM:0 | F1:0.61 这样的结构
    # 兼容整数和小数
    pattern = re.compile(r"EM:\s*([0-9\.]+)\s*\|\s*F1:\s*([0-9\.]+)")

    print(f"正在读取日志: {log_file_path} ...")
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                try:
                    em_val = float(match.group(1))
                    f1_val = float(match.group(2))
                    
                    em_total += em_val
                    f1_total += f1_val
                    count += 1
                except ValueError:
                    continue

    if count == 0:
        print("未在日志中找到任何有效的分数记录。")
    else:
        avg_em = em_total / count
        avg_f1 = f1_total / count
        
        print("-" * 30)
        print(f"已处理样本数: {count}")
        print(f"当前平均 EM : {avg_em:.4f} ({avg_em*100:.2f}%)")
        print(f"当前平均 F1 : {avg_f1:.4f} ({avg_f1*100:.2f}%)")
        print("-" * 30)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 如果用户没传参数，尝试读取用户刚才给的默认路径
        # default_path = "./logs/test/test.log" # 43 39.70
        # default_path = "./logs/llada_table_eval_128_checkpoint-400/eval_gpu0_lora_checkpoint-400_metrics.log"
        # default_path = "./logs/llada_table_eval_128_checkpoint-200/eval_gpu0_lora_checkpoint-200_metrics.log"
        # default_path = "./logs/llada_table_eval_128_checkpoint-600/eval_gpu0_lora_checkpoint-600_metrics.log"
        # default_path = "./logs/llada_table_eval_128_checkpoint-756/eval_gpu0_lora_checkpoint-756_metrics.log" # 43 41.35
        # default_path = "./logs/llada_eval_lora_1e-4_check-400_checkpoint-400/eval_gpu4_lora_checkpoint-400_metrics.log"
        default_path = "./logs/llada_eval_lora_1e-4_check-final_checkpoint-final/eval_gpu0_lora_checkpoint-final_metrics.log"

        # 已处理样本数: 82 0.4016

        if os.path.exists(default_path):
            calculate_scores(default_path)
        else:
            print("用法: python calc_score.py <日志文件路径>")
    else:
        calculate_scores(sys.argv[1])