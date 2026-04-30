import json
import os
import re
import time
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# ================= 配置区域 (请在此处修改) =================

# [1. API设置] 
# 在这里填入你的 DeepSeek API Key (或是中转商的 Key)
OPENAI_API_KEY = "sk-158c37af632346f3a0282902497ee983" 

# 如果是 DeepSeek 官方，填: "https://api.deepseek.com"
# 如果是用中转商(如 aigcbest)，保持原来的: "https://api2.aigcbest.top/v1"
OPENAI_BASE_URL = "https://api.deepseek.com" 

# [2. 模型设置]
# 切换为 DeepSeek-V3 (官方名称为 deepseek-chat)
# 如果用中转商，请确认中转商支持的模型名称
TEACHER_MODEL = "deepseek-chat" 

# [3. 文件路径]
INPUT_FILE = "../data/wikitql_train.jsonl"
OUTPUT_FILE = "../data/wikitql_train_cot.jsonl"

# [4. 采样数量]
# 设置为 None 表示跑全量
NUM_SAMPLES = None 

# [5. Prompt 设置]
SYSTEM_PROMPT = (
    "You are a concise table reasoning expert. "
    "Read the linearized table and the question. "
    "Provide a short, step-by-step reasoning process (1-3 sentences) to locate the answer. "
    "End your response strictly with the format: 'Answer: <final_entity>'."
)
# =======================================================

# 初始化客户端
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def normalize_answer(text):
    """归一化答案"""
    if not text: return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text) 
    return text

def verify_answer(model_output, gold_answers):
    """[阅卷老师] 校验函数"""
    match = re.search(r'answer:\s*(.*)', model_output, re.IGNORECASE)
    if not match: return False
    
    generated_ans = match.group(1).strip()
    norm_gen = normalize_answer(generated_ans)
    norm_golds = [normalize_answer(a) for a in gold_answers]
    
    if norm_gen in norm_golds: return True
    
    for gold in norm_golds:
        if gold and (gold in norm_gen or norm_gen in gold):
            if len(gold) > 1 or len(norm_gen) > 1: return True
            if gold == norm_gen: return True
    return False

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_with_retry(messages):
    """API 调用"""
    response = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=256,
        timeout=30
    )
    return response.choices[0].message.content.strip()

def process_line(line_str):
    """处理单行数据"""
    try:
        data = json.loads(line_str)
        user_content = data['messages'][0]['content']
        gold_answers = data['answers']
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        cot_content = generate_with_retry(messages)
        
        if verify_answer(cot_content, gold_answers):
            new_data = {
                "id": data['id'],
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": cot_content}
                ],
                "question": data.get("question", ""),
                "answers": gold_answers,
                "source": f"wtq_cot_filtered_{TEACHER_MODEL}"
            }
            return json.dumps(new_data, ensure_ascii=False)
        else:
            return None
    except Exception:
        return None

def main():
    print(f"[INFO] 任务启动 (ID断点续传模式)")
    print(f"输入文件: {INPUT_FILE}")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"当前模型: {TEACHER_MODEL}")
    
    if not os.path.exists(INPUT_FILE):
        print("[ERROR] 找不到输入文件")
        return

    # 1. 读取所有待处理的输入数据
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    # 2. [核心] 读取已完成的 ID 集合 (防止重复跑)
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        print("[INFO] 正在扫描已完成的数据...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if 'id' in record:
                        processed_ids.add(record['id'])
                except:
                    continue
    
    print(f"[INFO] 历史有效数据: {len(processed_ids)} 条")

    # 3. 过滤出真正需要跑的数据
    lines_to_process = []
    for line in all_lines:
        try:
            item = json.loads(line)
            # 只有当 ID 没在输出文件里出现过，才加入队列
            if item['id'] not in processed_ids:
                lines_to_process.append(line)
        except:
            continue

    # 采样限制
    if NUM_SAMPLES:
        remaining_quota = NUM_SAMPLES - len(processed_ids)
        if remaining_quota <= 0:
            print("[INFO] 已达到目标采样数，任务结束！")
            return
        lines_to_process = lines_to_process[:remaining_quota]

    print(f"[INFO] 本次计划处理: {len(lines_to_process)} 条")
    
    if not lines_to_process:
        print("[INFO] 没有新数据需要处理，休息一下吧~")
        return

    # 4. 多线程执行
    # DeepSeek 并发很高，这里建议开 20-50
    MAX_WORKERS = 20 
    
    success_count = 0
    total_processed = 0
    
    print(f"[INFO] 启动 {MAX_WORKERS} 个线程并发处理...")
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_line, line): line for line in lines_to_process}
            
            pbar = tqdm(concurrent.futures.as_completed(futures), total=len(lines_to_process))
            
            for future in pbar:
                result = future.result()
                total_processed += 1
                
                if result:
                    if not result.startswith("DEBUG"):
                        f_out.write(result + "\n")
                        f_out.flush()
                        success_count += 1
                
                # 计算本次运行的通过率
                current_acc = (success_count / total_processed) * 100 if total_processed > 0 else 0
                pbar.set_description(f"Valid: {success_count} ({current_acc:.1f}%)")

    print(f"\n[INFO] 本次任务结束！")
    print(f"总数据量 (含历史): {len(processed_ids) + success_count} 条")

if __name__ == "__main__":
    main()