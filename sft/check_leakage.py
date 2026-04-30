import json
import tqdm
import sys

# ================= 配置区域 =================
# 训练数据路径 (SFT的数据)
TRAIN_FILE = "data/llada_sft_structural_flash.jsonl"
# 测试数据路径 (你用来跑Eval的数据)
TEST_FILE = "data/tabfact_test.jsonl"
# ===========================================

def normalize_text(text):
    """
    标准化文本：
    1. 去除首尾空格
    2. 截断到 '[Answer]' 之前（防止因为结尾的空格或换行符不同导致误判）
    """
    if "[Answer]" in text:
        text = text.split("[Answer]")[0]
    return text.strip()

def load_dataset_fingerprints(file_path, is_train=True):
    fingerprints = set()
    table_fingerprints = set()
    count = 0
    
    print(f"Loading {'Train' if is_train else 'Test'} data from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f):
            try:
                data = json.loads(line)
                # 提取 User Prompt
                if "messages" in data:
                    content = data["messages"][0]["content"]
                elif "prompt" in data: # 兼容其他格式
                    content = data["prompt"]
                else:
                    continue
                
                # 1. 提取核心内容 (Table + Statement)
                core_text = normalize_text(content)
                
                if is_train:
                    fingerprints.add(core_text)
                    
                    # 2. 尝试提取表格指纹 (Table Only)
                    # 假设格式是 ... [Reference Data] ... [Statement] ...
                    if "[Reference Data]" in core_text and "[Statement]" in core_text:
                        table_part = core_text.split("[Statement]")[0].strip()
                        table_fingerprints.add(table_part)
                else:
                    # 测试集返回列表，方便逐个检查
                    fingerprints.add(core_text)
                    
                    if "[Reference Data]" in core_text and "[Statement]" in core_text:
                        table_part = core_text.split("[Statement]")[0].strip()
                        table_fingerprints.add(table_part)
                        
                count += 1
            except Exception as e:
                print(f"Error reading line: {e}")
                
    print(f"Loaded {count} samples.")
    return fingerprints, table_fingerprints

def main():
    print("=== 🔍 开始数据泄漏检测 ===")
    
    # 1. 加载训练集指纹 (存入哈希表)
    train_prompts, train_tables = load_dataset_fingerprints(TRAIN_FILE, is_train=True)
    
    # 2. 检查测试集
    test_prompts, test_tables = load_dataset_fingerprints(TEST_FILE, is_train=False)
    
    print("\n=== 📊 检测结果 ===")
    
    # --- 检测 1: 严格泄漏 (Statement + Table 完全一致) ---
    # 也就是“这道题老师上课讲过原题”
    exact_leaks = 0
    for t_prompt in test_prompts:
        if t_prompt in train_prompts:
            exact_leaks += 1
            
    # --- 检测 2: 表格泄漏 (Table 一致，但 Statement 可能不同) ---
    # 也就是“这个表格背景资料上课见过”
    table_leaks = 0
    for t_table in test_tables:
        if t_table in train_tables:
            table_leaks += 1
            
    total_test = len(test_prompts)
    
    print(f"\nTotal Test Samples: {total_test}")
    
    print("-" * 40)
    print(f"🔴 Exact Leakage (Statement Level): {exact_leaks} / {total_test}")
    print(f"   Leakage Rate: {exact_leaks / total_test * 100:.4f}%")
    if exact_leaks == 0:
        print("   ✅ PASS! 没有发现题目泄漏！")
    else:
        print("   ❌ WARNING! 发现原题泄漏！分数不可信！")
        
    print("-" * 40)
    print(f"🟡 Table Leakage (Table Level):     {table_leaks} / {len(test_tables)}")
    print(f"   Leakage Rate: {table_leaks / len(test_tables) * 100:.4f}%")
    if table_leaks == 0:
        print("   ✅ Clean Split! 测试集的表格从未在训练集中出现过。")
    else:
        print("   ⚠️ Note: 部分表格在训练集中出现过（TabFact 标准划分允许这种情况吗？通常需要确认）。")

if __name__ == "__main__":
    main()