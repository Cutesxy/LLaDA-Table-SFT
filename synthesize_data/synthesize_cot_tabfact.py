import json
import os
import re
import hashlib
import random
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# ================= 配置区域 =================

# [1. API设置] 
OPENAI_API_KEY = "sk-158c37af632346f3a0282902497ee983" 
OPENAI_BASE_URL = "https://api.deepseek.com" 

# [2. 模型设置]
TEACHER_MODEL = "deepseek-chat" 

# [3. 文件路径]
INPUT_FILE = "../data/tab_fact_train.jsonl" 
OUTPUT_FILE = "../data/tab_fact_train_cot.jsonl"

# [4. 目标有效数量]
TARGET_COUNT = 1000 

# [5. Prompt 设置]
SYSTEM_PROMPT = (
    "You are a concise table fact-checking expert. "
    "Read the linearized table and the statement. "
    "Provide a short reasoning process (1-3 sentences) to verify if the statement is supported by the table. "
    "End your response strictly with: 'Answer: entailed' or 'Answer: refuted'."
)

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def format_tabfact_table(raw_text):
    """格式化表格"""
    try:
        lines = raw_text.strip().split('\n')
        if not lines: return ""
        header = lines[0].replace('#', ' | ')
        formatted = f"[TAB] col: | {header} |"
        for line in lines[1:]:
            row = line.replace('#', ' | ')
            formatted += f" [SEP] | {row} |"
        return formatted
    except:
        return raw_text

def verify_tabfact(model_output, gold_label):
    """校验逻辑"""
    match = re.search(r'answer:\s*(entailed|refuted)', model_output, re.IGNORECASE)
    if not match: return False
    
    pred_str = match.group(1).lower()
    if gold_label == 1 and pred_str == 'entailed': return True
    if gold_label == 0 and pred_str == 'refuted': return True
    return False

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_with_retry(messages):
    response = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=200, 
        timeout=30
    )
    return response.choices[0].message.content.strip()

def process_line(line_str):
    try:
        data = json.loads(line_str)
        raw_table = data.get('table_text', '')
        caption = data.get('table_caption', '')
        statement = data.get('statement', '')
        gold_label = data.get('label')
        
        # 简单长度过滤
        if len(raw_table) > 12000: return None

        linear_table = format_tabfact_table(raw_table)
        
        user_content = (
            f"Table Caption: {caption}\n\n"
            f"{linear_table}\n\n"
            f"Statement: {statement}"
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        cot_content = generate_with_retry(messages)
        
        if verify_tabfact(cot_content, gold_label):
            unique_hash = hashlib.md5(statement.encode('utf-8')).hexdigest()[:8]
            new_data = {
                "id": f"tabfact_{unique_hash}", 
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": cot_content}
                ],
                "label": gold_label,
                "source": f"tabfact_cot_{TEACHER_MODEL}"
            }
            return json.dumps(new_data, ensure_ascii=False)
        else:
            return None
    except:
        return None

def main():
    print(f"[INFO] TabFact 随机合成任务启动")
    print(f"输入: {INPUT_FILE}")
    print(f"输出: {OUTPUT_FILE}")
    print(f"目标数量: {TARGET_COUNT}")
    
    if not os.path.exists(INPUT_FILE):
        print("[ERROR] 找不到输入文件")
        return

    # 1. 读取所有数据
    print("[INFO] 正在读取并加载所有原始数据...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    print(f"[INFO] 原始数据总数: {len(all_lines)}")

    # 2. 检查已完成进度
    existing_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for _ in f: existing_count += 1
    print(f"[INFO] 历史已完成: {existing_count} 条")
    
    if existing_count >= TARGET_COUNT:
        print("[INFO] 目标已达成，无需运行。")
        return

    # 3. 随机采样策略
    needed_valid = TARGET_COUNT - existing_count
    needed_input = int(needed_valid * 1.4)
    
    print(f"[INFO] 还需要 {needed_valid} 条有效数据，正在随机抽取 {needed_input} 条样本...")

    random.seed(42) 
    random.shuffle(all_lines)
    
    lines_to_process = all_lines[:needed_input]

    # 4. 并发执行
    MAX_WORKERS = 20
    success_count = 0
    total_processed = 0

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_line, line): line for line in lines_to_process}
            pbar = tqdm(concurrent.futures.as_completed(futures), total=len(lines_to_process))
            
            for future in pbar:
                result = future.result()
                total_processed += 1
                if result:
                    f_out.write(result + "\n")
                    f_out.flush()
                    success_count += 1
                    
                    if (existing_count + success_count) >= TARGET_COUNT:
                        print(f"\n[INFO] 目标达成 ({TARGET_COUNT})，停止任务！")
                        os._exit(0)
                
                acc = (success_count / total_processed) * 100 if total_processed else 0
                pbar.set_description(f"Total Valid: {existing_count + success_count}/{TARGET_COUNT} | Rate: {acc:.1f}%")

    print(f"\n[INFO] 任务结束！")

if __name__ == "__main__":
    main()