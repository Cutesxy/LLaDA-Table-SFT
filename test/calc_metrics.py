import os
import re
import glob
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Average F1/EM from Logs")
    # 默认指向你刚才展示的目录
    parser.add_argument('--log_dir', type=str, default='../logs/llada_table_eval_128_checkpoint-400', 
                        help='Path to the log directory containing eval_gpu*_metrics.log')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 寻找目录下所有的 gpu metrics 日志
    search_pattern = os.path.join(args.log_dir, "eval_gpu*_metrics.log")
    log_files = sorted(glob.glob(search_pattern))
    
    if not log_files:
        print(f"Error: No log files found in {args.log_dir}")
        return

    print(f"Found {len(log_files)} log files in '{args.log_dir}'\n")

    # 全局统计变量
    global_f1_scores = []
    global_em_scores = []
    
    # 正则表达式匹配：EM:0 | F1:0.23
    # 这种写法比较稳健，只要行里有 F1:数字 就能抓出来
    pattern = re.compile(r"EM:(\d+)\s*\|\s*F1:([\d\.]+)")

    print(f"{'Log File':<30} | {'Samples':<8} | {'Avg F1':<10} | {'Avg EM':<10}")
    print("-" * 70)
    
    for log_file in log_files:
        file_f1 = []
        file_em = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        em_val = int(match.group(1))
                        f1_val = float(match.group(2))
                        
                        file_f1.append(f1_val)
                        file_em.append(em_val)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            continue

        # 将当前文件的结果加入全局列表
        global_f1_scores.extend(file_f1)
        global_em_scores.extend(file_em)
        
        # 打印当前文件的统计
        filename = os.path.basename(log_file)
        if len(file_f1) > 0:
            print(f"{filename:<30} | {len(file_f1):<8} | {np.mean(file_f1)*100:.2f}%     | {np.mean(file_em)*100:.2f}%")
        else:
            print(f"{filename:<30} | 0        | N/A        | N/A")

    print("-" * 70)

    # 打印全局汇总
    total_samples = len(global_f1_scores)
    if total_samples > 0:
        avg_f1 = np.mean(global_f1_scores)
        avg_em = np.mean(global_em_scores)
        
        print(f"\nFINAL SUMMARY:")
        print(f"   Total Samples: {total_samples}")
        print(f"   Global Avg F1: {avg_f1 * 100:.4f}%")
        print(f"   Global Avg EM: {avg_em * 100:.4f}%")
    else:
        print("\nNo valid metrics found in any logs.")

if __name__ == "__main__":
    main()