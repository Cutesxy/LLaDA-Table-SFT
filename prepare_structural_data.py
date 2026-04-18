import json
import random
import os
from tqdm import tqdm

# ================= 配置区域 =================
# 原始数据路径
TABFACT_FILE = "data/tab_fact_train.jsonl"
WTQ_FILE = "data/wikitql_train.jsonl"

# 输出文件
OUTPUT_FILE = "data/llada_sft_mixed_train.jsonl"

# --- 策略选择 ---
# 模式 A: 'debug' (各取 14k，快速验证代码通不通)
# 模式 B: 'sota'  (TabFact 全留，WTQ 复制 N 遍以平衡数量，推荐！)
MODE = 'sota' 

# 仅在 debug 模式下生效
DEBUG_LIMIT = 14000 

def format_tabfact_table(table_text):
    """处理 TabFact 表格: 用 # 分隔 -> 标准线性格式"""
    if not table_text: return ""
    rows = table_text.strip().split('\n')
    formatted_rows = ["| " + " | ".join([c.strip() for c in r.split('#')]) + " |" for r in rows]
    return " [SEP] ".join(formatted_rows)

def format_wtq_table(table_text):
    """
    处理 WTQ 表格: 
    输入通常是: "[TAB] col: | year | ... | [SEP] | 2001 | ..."
    我们要去掉开头的标记，确保和 TabFact 格式一致 (Table LLaDA 只能适应一种表格视觉)
    """
    if not table_text: return ""
    # 去掉可能存在的特定前缀，保持纯净
    clean_text = table_text.replace("[TAB] col: ", "").replace("[TAB]", "").strip()
    return clean_text

def process_tabfact(file_path):
    samples = []
    print(f"Loading TabFact from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            label = data.get('label', -1)
            if label not in [0, 1]: continue
            
            target = "entailed" if label == 1 else "refuted"
            table_str = format_tabfact_table(data.get('table_text', ''))
            statement = data.get('statement', '')
            
            # 构造 TabFact Prompt
            user_prompt = (
                "== Table Verification Task ==\n\n"
                "[Reference Data]\n"
                f"{table_str}\n\n"
                "-------------------\n\n"
                "[Statement]\n"
                f"{statement}\n\n"
                "[Answer]\n"
            )
            
            samples.append({
                "source": "tabfact",
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": target}
                ]
            })
    return samples

def process_wtq(file_path):
    samples = []
    print(f"Loading WTQ from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            
            # WTQ 的 table_text 已经在 json 里是线性格式了，只需要微调
            table_str = format_wtq_table(data.get('table_text', ''))
            question = data.get('question', '')
            
            # 处理答案：WTQ 有多个别名，训练时我们通常取第一个主要答案
            # 你的 SFT 代码会自动加 EOS，所以这里只需要纯文本
            answers = data.get('answers', [])
            if not answers: continue
            target = answers[0] 
            
            # 构造 WTQ Prompt (结构化填空风格)
            user_prompt = (
                "== Table Analysis Task ==\n\n"
                "[Reference Data]\n"
                f"{table_str}\n\n"
                "-------------------\n\n"
                "[Question]\n"
                f"{question}\n\n"
                "[Answer]\n"
            )
            
            samples.append({
                "source": "wtq",
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": target}
                ]
            })
    return samples

def main():
    # 1. 读取数据
    tabfact_data = process_tabfact(TABFACT_FILE)
    wtq_data = process_wtq(WTQ_FILE)
    
    print(f"原始数据量 -> TabFact: {len(tabfact_data)}, WTQ: {len(wtq_data)}")
    
    final_data = []
    
    # 2. 根据策略混合
    if MODE == 'debug':
        print(f"⚠️ 模式: DEBUG (各取 {DEBUG_LIMIT} 条)")
        random.shuffle(tabfact_data)
        random.shuffle(wtq_data)
        final_data = tabfact_data[:DEBUG_LIMIT] + wtq_data[:DEBUG_LIMIT]
        
    elif MODE == 'sota':
        print("🔥 模式: SOTA (保留全量 TabFact，过采样 WTQ)")
        
        # 计算 WTQ 需要复制多少倍才能赶上 TabFact
        upsample_ratio = len(tabfact_data) // len(wtq_data)
        # 限制最大倍数，防止过拟合太严重 (比如复制 5-6 倍即可)
        upsample_ratio = min(upsample_ratio, 6) 
        
        print(f"WTQ 数据将被复制 {upsample_ratio} 倍...")
        
        final_data = tabfact_data + (wtq_data * upsample_ratio)
        
    # 3. 全局打乱
    random.shuffle(final_data)
    
    # 4. 保存
    print(f"正在写入 {OUTPUT_FILE} (共 {len(final_data)} 条)...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item) + "\n")
            
    print("✅ 混合数据准备完毕！")

if __name__ == "__main__":
    main()