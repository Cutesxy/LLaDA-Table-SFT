import json
import os
import sys
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# ================= 配置区域 =================
# 指向你的数据文件（相对路径：上级目录 data 下）
DATA_FILE = "../../data/table_llada_train.jsonl"

# 模型路径（用于加载 Tokenizer，确保计算准确）
# 如果你本地有模型，请修改为本地绝对路径；否则使用 HuggingFace ID
MODEL_PATH = "/home/zjusst/hxy/llada/models/GSAI-ML/LLaDA-8B-Instruct" 

# 设定的最大长度
MAX_LENGTH = 4096
# ===========================================

def load_jsonl(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件 {file_path}")
        sys.exit(1)
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def main():
    print(f"🔄 正在加载 Tokenizer: {MODEL_PATH} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"❌ 加载 Tokenizer 失败: {e}")
        print("请检查网络或将 MODEL_PATH 修改为本地已下载的模型路径。")
        sys.exit(1)

    print(f"📖 正在读取数据: {DATA_FILE} ...")
    data = load_jsonl(DATA_FILE)
    total_count = len(data)
    print(f"✅ 总共有 {total_count} 条数据。开始分析长度...")

    # 统计容器
    stats = {
        "safe": 0,          # 长度 <= 4096 (安全)
        "discard_filter": 0, # Prompt 本身 > 4096 (如果开Filter会被扔掉)
        "danger_truncate": 0, # Prompt <= 4096 但总长 > 4096 (Answer会被切断)
        "lengths_total": [],  # 记录所有总长度
        "lengths_prompt": []  # 记录所有 Prompt 长度
    }

    # 进度条遍历
    for i, sample in tqdm(enumerate(data), total=total_count):
        messages = sample.get("messages", [])
        
        # 1. 提取 User (Context) 和 Assistant (Answer)
        user_content = ""
        assistant_content = ""
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                user_content += content + "\n" # 通常拼接会有换行
            elif role == "assistant":
                assistant_content += content
        
        # 2. 计算 Token 长度 (模拟实际训练时的编码)
        # 注意：这里我们分别编码再相加，误差极小（可能有1-2个特殊token的偏差）
        # add_special_tokens=False 是为了得到纯文本的长度
        p_ids = tokenizer.encode(user_content, add_special_tokens=False)
        a_ids = tokenizer.encode(assistant_content, add_special_tokens=False)
        
        p_len = len(p_ids)
        a_len = len(a_ids)
        total_len = p_len + a_len + 2 # +2 是预估 BOS 和 EOS 的开销

        # 记录分布
        stats["lengths_total"].append(total_len)
        stats["lengths_prompt"].append(p_len)

        # 3. 判定分类
        if p_len > MAX_LENGTH:
            # 连题目都放不下
            stats["discard_filter"] += 1
        elif total_len > MAX_LENGTH:
            # 题目放得下，但加上答案超了 -> 隐患数据！
            stats["danger_truncate"] += 1
        else:
            # 安全数据
            stats["safe"] += 1

    # ================= 输出详细报告 =================
    lengths_arr = np.array(stats["lengths_total"])
    
    print("\n" + "="*60)
    print(f"📊 数据长度分析报告 (Max Length = {MAX_LENGTH})")
    print("="*60)
    
    print(f"1. ✅ 安全数据 (Safe):")
    print(f"   数量: {stats['safe']} / {total_count} ({stats['safe']/total_count*100:.2f}%)")
    print(f"   说明: 完整保留，无需截断。")
    print("-" * 40)
    
    print(f"2. 🗑️  建议丢弃 (Discard - Prompt太长):")
    print(f"   数量: {stats['discard_filter']} / {total_count} ({stats['discard_filter']/total_count*100:.2f}%)")
    print(f"   说明: 表格太大了，连题目都读不完。开启 Filter 后这些会被删掉。")
    print("-" * 40)
    
    print(f"3. ⚠️  截断隐患 (Danger - Answer被切):")
    print(f"   数量: {stats['danger_truncate']} / {total_count} ({stats['danger_truncate']/total_count*100:.2f}%)")
    print(f"   说明: 这里的答案会被切断！如果用 truncation='right'，这些就是“烂尾”数据。")
    print(f"   建议: 如果这个比例很高，请务必开启 truncation='filter'。")
    print("="*60)

    # 长度分布统计
    print(f"📈 长度分布统计 (Tokens):")
    print(f"   平均长度: {np.mean(lengths_arr):.1f}")
    print(f"   中位数 (P50): {np.median(lengths_arr):.1f}")
    print(f"   P90 (90%的数据小于): {np.percentile(lengths_arr, 90):.1f}")
    print(f"   P95 (95%的数据小于): {np.percentile(lengths_arr, 95):.1f}")
    print(f"   最大长度: {np.max(lengths_arr)}")
    print("="*60)
    
    print("\n💡 建议操作：")
    if stats['danger_truncate'] > 0:
        print(f"检测到 {stats['danger_truncate']} 条数据会导致答案截断。")
        print("请在训练脚本中添加: --truncation filter")
    else:
        print("数据非常完美，没有截断风险。")

if __name__ == "__main__":
    main()