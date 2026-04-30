import os
import json
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI

# ================= 配置区域 (请修改这里) =================
# 1. 你的 API Key
API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 

# 2. API Base URL (如果是官方就不用改，如果是中转请改为中转地址)
BASE_URL = "https://api.openai.com/v1" 

# 3. 你的日志目录 (根据你截图的路径)
LOG_DIR = "logs/eval_wtq_BaseInstruct_step32_len32"

# 4. 评估模型
JUDGE_MODEL = "gpt-4o"

# 5. 并发线程数 (根据你的额度调整，建议 10-20)
MAX_WORKERS = 10
# =======================================================

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_judge_prompt(ground_truth, prediction):
    """
    【极简版 Prompt】
    依靠 GPT-4o 的强大直觉，判断预测是否正确。
    忽略格式废话，但逻辑必须一致。
    """
    return f"""
System: You are a helpful evaluator for a QA task.

Task: Determine if the [Model Prediction] matches the [Ground Truth] answer.

[Ground Truth]: "{ground_truth}"
[Model Prediction]: "{prediction}"

Judge: Does the prediction provide the correct answer? (Ignore minor formatting differences or extra words like "The answer is", but ensure the meaning is accurate).

Answer only "YES" or "NO".
"""

def judge_single_case(item):
    """
    评估单个样本
    """
    # 数据清洗，防止 None 报错
    gt = str(item.get("ground_truth", "")).strip()
    pred = str(item.get("prediction", "")).strip()
    
    # 1. 省钱策略：完全字符串匹配 (大小写不敏感)
    # 如果完全一样，肯定是 YES，不需要问 GPT-4o
    if gt.lower() == pred.lower():
        return True, item
    
    # 特殊处理：如果是数字，尝试转 float 比较 (如 5.0 vs 5)
    try:
        if float(gt.replace(',', '')) == float(pred.replace(',', '')):
            return True, item
    except ValueError:
        pass

    # 2. 调用 GPT-4o 进行判断
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "user", "content": get_judge_prompt(gt, pred)}
            ],
            temperature=0.0, # 只需要确定性输出
            max_tokens=5     # 只需输出 YES/NO
        )
        result = response.choices[0].message.content.strip().upper()
        
        # 只要包含 YES 就算对 (处理可能出现的 "YES." 等情况)
        is_correct = "YES" in result
        return is_correct, item
        
    except Exception as e:
        print(f"API Error: {e}")
        return False, item

def load_all_cases(directory):
    """
    读取目录下所有的 eval_gpu*_cases.jsonl 文件
    """
    # 匹配模式：eval_gpu*_cases.jsonl
    pattern = os.path.join(directory, "*_cases.jsonl")
    all_files = glob.glob(pattern)
    all_data = []
    
    print(f"正在读取文件: {all_files}")
    
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_data.append(json.loads(line))
        except Exception as e:
            print(f"读取文件错误 {file_path}: {e}")
            
    return all_data

def main():
    # 1. 加载数据
    if not os.path.exists(LOG_DIR):
        print(f"错误：目录 {LOG_DIR} 不存在！")
        return

    data = load_all_cases(LOG_DIR)
    if not data:
        print("未找到任何数据，请检查文件名是否包含 '_cases.jsonl'")
        return

    print(f"共加载 {len(data)} 条样本，开始 GPT-4o 评估...")

    # 2. 并发评估
    correct_count = 0
    total_count = 0
    results = []

    # 使用 tqdm 显示进度条
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(judge_single_case, item): item for item in data}
        
        for future in tqdm(futures, total=len(data), desc="Evaluating"):
            is_correct, original_item = future.result()
            
            total_count += 1
            if is_correct:
                correct_count += 1
            
            # 记录结果，方便复盘
            results.append({
                "ground_truth": original_item.get("ground_truth"),
                "prediction": original_item.get("prediction"),
                "judge_correct": is_correct
            })

    # 3. 输出报告
    if total_count == 0:
        print("没有有效样本被评估。")
        return

    acc = (correct_count / total_count) * 100
    
    print("\n" + "="*50)
    print(f"评估报告 (Judge: {JUDGE_MODEL})")
    print("="*50)
    print(f"Total Samples : {total_count}")
    print(f"Correct       : {correct_count}")
    print(f"Accuracy      : {acc:.2f}%")
    print("="*50)

    # 4. 保存详细结果
    output_path = os.path.join(LOG_DIR, "gpt4o_eval_results_full.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"详细结果已保存至: {output_path}")

if __name__ == "__main__":
    main()