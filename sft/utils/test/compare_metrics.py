import os
import re
import glob
import datetime

# ================= 配置区域 =================
LOG_ROOT = "../../logs"
OUTPUT_DIR = "../../logs/analyze"

# 文件夹列表
TARGET_FOLDERS = [
    ("Base",    "llada_table_eval_128"),                
    ("Ckpt200", "llada_table_eval_128_checkpoint-200"), 
    ("Ckpt400", "llada_table_eval_128_checkpoint-400"), 
    ("Ckpt600", "llada_table_eval_128_checkpoint-600")
    
]

GPU_IDS = ['0', '1', '2']
SAMPLES_PER_GPU = 20 
# ===========================================

def strip_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def get_gpu_specific_data(folder_name):
    """
    读取数据，返回结构: { '0': {1: score, ...}, '1': {...}, '2': {...} }
    """
    # 初始化嵌套字典
    gpu_data = {gid: {} for gid in GPU_IDS}
    
    for gpu_id in GPU_IDS:
        search_pattern = os.path.join(LOG_ROOT, folder_name, f"eval_gpu{gpu_id}*metrics.log")
        found_files = glob.glob(search_pattern)
        
        if not found_files:
            continue
            
        log_path = found_files[0]
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                count = 0
                for line in f:
                    match = re.search(r"\[(\d+)\] .*?F1:(\d+\.\d+)", line)
                    if match:
                        idx = int(match.group(1))
                        f1 = float(match.group(2))
                        
                        # 存入对应 GPU 的字典中
                        gpu_data[gpu_id][idx] = f1
                        
                        count += 1
                        if count >= SAMPLES_PER_GPU:
                            break
        except Exception as e:
            print(f"[ERROR] Reading {log_path}: {e}")
            
    return gpu_data

def get_trend_icon(scores):
    valid_scores = [s for s in scores if s is not None]
    if len(valid_scores) < 2: return ""
    if all(y > x for x, y in zip(valid_scores, valid_scores[1:])): return "\033[92m↑\033[0m"
    if all(y < x for x, y in zip(valid_scores, valid_scores[1:])): return "\033[91m↓\033[0m"
    return "〰️"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"f1_trend_report_{timestamp}.txt"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    f_out = open(output_path, 'w', encoding='utf-8')

    def tee_print(text):
        print(text)
        f_out.write(strip_ansi_codes(text) + "\n")

    tee_print(f"Analysis Time: {timestamp}")
    tee_print(f"Loading top {SAMPLES_PER_GPU} samples from EACH of GPUs {GPU_IDS} (Total {len(GPU_IDS)*SAMPLES_PER_GPU})...\n")
    
    # 1. 读取所有数据
    # 结构: [ {'0':{...}, '1':{...}},  {'0':{...}, '1':{...}}, ... ]
    all_model_data = [] 
    headers = ["ID"]

    for label, folder in TARGET_FOLDERS:
        headers.append(label)
        data = get_gpu_specific_data(folder)
        all_model_data.append(data)

    # 2. 打印表头
    # ID列加宽一点，因为要显示 'G0-#12'
    header_str = f"{headers[0]:<9} | " + " | ".join([f"{h:<8}" for h in headers[1:]]) + " | Trend"
    tee_print("-" * (len(header_str) + 5))
    tee_print(header_str)
    tee_print("-" * (len(header_str) + 5))

    # 3. 三重循环打印：GPU -> Local ID
    total_printed = 0
    
    for gpu_id in GPU_IDS:
        # 打印分隔符，区分 GPU
        if total_printed > 0:
             tee_print(f"{'---':<9} | " + " | ".join(["---" for _ in headers[1:]]) + " |")

        for i in range(1, SAMPLES_PER_GPU + 1):
            # 构造显示的 ID，例如 "G0-#1"
            display_id = f"G{gpu_id}-#{i}"
            
            row_scores = []
            row_str = f"{display_id:<9} | "
            
            for model_idx, model_data in enumerate(all_model_data):
                # model_data 是 { '0': {...}, '1': {...} }
                # 取出当前 GPU 的数据字典
                current_gpu_dict = model_data.get(gpu_id, {})
                
                if not current_gpu_dict and not model_data: 
                    # 整个模型文件夹没读到
                    row_scores.append(None)
                    row_str += f"{'ERR':<8} | "
                elif i in current_gpu_dict:
                    score = current_gpu_dict[i]
                    row_scores.append(score)
                    row_str += f"{score:.2f}     | "
                else:
                    # 读到了文件，但该文件里没有这个 ID (可能文件不够长)
                    row_scores.append(None)
                    row_str += f"{'Miss':<8} | "
            
            tee_print(f"{row_str}{get_trend_icon(row_scores)}")
            total_printed += 1

    tee_print("-" * (len(header_str) + 5))
    tee_print(f"Total samples displayed: {total_printed}")
    tee_print(f"\n[Done] Report saved to: {output_path}")
    
    f_out.close()

if __name__ == "__main__":
    main()