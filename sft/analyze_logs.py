import json
import os
import glob
import numpy as np

def analyze_cot_performance(log_dir):
    # 1. 获取目录下所有的 json/jsonl 文件
    # 假设你的日志文件后缀是 .json 或 .jsonl
    log_files = glob.glob(os.path.join(log_dir, "*.json")) + glob.glob(os.path.join(log_dir, "*.jsonl"))
    
    if not log_files:
        print(f"❌ Error: No json/jsonl files found in directory: {log_dir}")
        return

    print(f"📂 Found {len(log_files)} log files in {log_dir}...")

    # 2. 初始化统计容器
    stats = {
        "total": 0,
        "cot": {"count": 0, "correct": 0},
        "direct": {"count": 0, "correct": 0}
    }

    # 3. 逐行读取并分类
    for file_path in log_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                stats["total"] += 1
                
                # 获取关键信息
                raw_output = data.get("raw_output", "").strip()
                is_correct = data.get("metrics", {}).get("em", 0)

                # =================================================
                # 核心分类逻辑 (Classification Logic)
                # =================================================
                # 判别标准：
                # 1. 如果以 "Answer:" 开头 -> Direct
                # 2. 如果以 "Step 2" 开头 (跳过了 Step 1) -> Direct
                # 3. 如果包含 "Step 1" 或 "Analysis" -> CoT
                # =================================================
                is_cot = False
                lower_output = raw_output.lower()

                if lower_output.startswith("answer:") or lower_output.startswith("step 2"):
                    is_cot = False
                elif "step 1" in lower_output or "analysis" in lower_output:
                    is_cot = True
                else:
                    # 兜底：如果很短且没有明显的推理词，算 Direct；否则算 CoT
                    if len(raw_output) < 100:
                        is_cot = False
                    else:
                        is_cot = True

                # 4. 统计归档
                key = "cot" if is_cot else "direct"
                stats[key]["count"] += 1
                if is_correct:
                    stats[key]["correct"] += 1

    # 4. 计算并打印结果
    print("\n" + "="*50)
    print(f"📊 Analysis Result for: {log_dir}")
    print("="*50)
    
    total_samples = stats["total"]
    if total_samples == 0:
        print("No data found.")
        return

    # 计算 Direct 组数据
    direct_cnt = stats["direct"]["count"]
    direct_acc = (stats["direct"]["correct"] / direct_cnt * 100) if direct_cnt > 0 else 0.0
    direct_ratio = (direct_cnt / total_samples * 100)

    # 计算 CoT 组数据
    cot_cnt = stats["cot"]["count"]
    cot_acc = (stats["cot"]["correct"] / cot_cnt * 100) if cot_cnt > 0 else 0.0
    cot_ratio = (cot_cnt / total_samples * 100)

    # 打印报表
    print(f"Total Samples: {total_samples}")
    print("-" * 50)
    
    print(f"🔹 [Direct Answer] (Skipped Analysis)")
    print(f"   - Count:    {direct_cnt} ({direct_ratio:.1f}%)")
    print(f"   - Accuracy: {direct_acc:.2f}%")
    print("-" * 50)
    
    print(f"🔸 [Chain of Thought] (Triggered Step 1)")
    print(f"   - Count:    {cot_cnt} ({cot_ratio:.1f}%)")
    print(f"   - Accuracy: {cot_acc:.2f}%")
    print("=" * 50)

    # 5. 简单的结论推导
    print("\n🔍 Insight Analysis:")
    if cot_acc > direct_acc:
        print(f"✅ CoT works! Reasoning improves accuracy by +{cot_acc - direct_acc:.2f}%.")
        print("   The model correctly identifies harder problems and solves them via reasoning.")
    elif cot_acc < direct_acc:
        print(f"⚠️ CoT Tax observed. Reasoning reduces accuracy by -{direct_acc - cot_acc:.2f}%.")
        print("   The model might be hallucinating during reasoning, or Direct questions are just easier.")
    else:
        print("   CoT and Direct performance are similar.")

    print(f"   Trigger Rate: {cot_ratio:.1f}% of questions triggered reasoning.")

# =================================================
# 使用方法：修改这里的路径为你实际的日志目录
# =================================================
if __name__ == "__main__":
    # 这里填你刚才说的日志路径
    LOG_DIRECTORY = "./logs/wtq_cot_llada_eval_v1" 
    
    analyze_cot_performance(LOG_DIRECTORY)