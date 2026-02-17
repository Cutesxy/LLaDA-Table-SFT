import json
import os

file_path = "./data/feverous_test.jsonl"

stats = {
    "supports": 0,
    "refutes": 0,
    "not enough info": 0,
    "other": 0
}
total_count = 0

print(f"正在分析文件: {file_path} ...")

try:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                total_count += 1
                
                # 提取 Label
                label = None
                
                # 优先从 messages 里找 assistant 的回答
                if "messages" in data:
                    for msg in data["messages"]:
                        if msg["role"] == "assistant":
                            label = msg["content"].strip().lower()
                            break
                
                # 如果 messages 里没找到，尝试找 label 字段 (兜底)
                if not label and "label" in data:
                    label = str(data["label"]).strip().lower()
                
                # 统计逻辑
                if label:
                    if "supports" in label:
                        stats["supports"] += 1
                    elif "refutes" in label:
                        stats["refutes"] += 1
                    elif "not enough" in label:
                        stats["not enough info"] += 1
                    else:
                        stats["other"] += 1
                        # 可以取消注释下面这行来看看是什么奇怪的标签
                        # print(f"Unknown label: {label}")
                else:
                    stats["other"] += 1

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line")

    # === 输出结果 ===
    print("-" * 30)
    print(f"【统计结果】 总样本数: {total_count}")
    print("-" * 30)
    
    if total_count > 0:
        print(f"Supports        : {stats['supports']:<5} ({stats['supports']/total_count*100:.2f}%)")
        print(f"Refutes         : {stats['refutes']:<5} ({stats['refutes']/total_count*100:.2f}%)")
        print(f"Not Enough Info : {stats['not enough info']:<5} ({stats['not enough info']/total_count*100:.2f}%)")
        print(f"Other/Unknown   : {stats['other']:<5} ({stats['other']/total_count*100:.2f}%)")
    print("-" * 30)

    # === 关键判定 ===
    if stats['not enough info'] == 0:
        print("✅ 结论：这是一个纯二分类 (Binary) 数据集。你的 90% 是实打实的 SOTA！")
    else:
        print("⚠️ 注意：数据集中包含 'Not Enough Info'。")
        print("    如果你的评测代码把它强制转为 Refuted，分数可能存在偏差。")

except FileNotFoundError:
    print(f"错误: 找不到文件 {file_path}")