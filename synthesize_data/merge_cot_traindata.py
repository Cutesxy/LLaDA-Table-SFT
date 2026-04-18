import json
import random
import os
from tqdm import tqdm

# ================= 配置 =================
INPUT_FILES = [
    "../data/wikitql_train_cot.jsonl",      # WTQ 数据
    "../data/tab_fact_train_cot.jsonl"      # TabFact 数据
]

OUTPUT_FILE = "../data/llada_sft_final_train.jsonl"
# =======================================

def main():
    print("[INFO] 开始合并任务...")
    
    all_data = []
    
    # 1. 读取文件
    for file_path in INPUT_FILES:
        if not os.path.exists(file_path):
            print(f"[WARN] 文件不存在: {file_path}")
            continue
            
        print(f"[INFO] 正在处理: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    item = json.loads(line)
                    
                    if "messages" in item and item["messages"]:
                        # 只提取 messages 字段
                        clean_item = {
                            "messages": item["messages"]
                        }
                        all_data.append(clean_item)
                        
                except json.JSONDecodeError:
                    continue

    print(f"[INFO] 读取完成，总数据量: {len(all_data)}")

    # 2. 随机打乱
    print("[INFO] 正在随机打乱数据...")
    random.seed(42)
    random.shuffle(all_data)

    # 3. 写入文件
    print(f"[INFO] 正在写入: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in tqdm(all_data):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[INFO] 任务完成")
    print(f"最终文件路径: {OUTPUT_FILE}")
    print(f"最终行数: {len(all_data)}")

if __name__ == "__main__":
    main()