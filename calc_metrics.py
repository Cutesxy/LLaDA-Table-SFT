import os
import json
import glob

def calculate_metrics(log_dir):
    # 1. 寻找目录下所有的 *cases.jsonl 文件
    # 使用 os.path.join 确保路径拼接正确
    search_pattern = os.path.join(log_dir, "*cases.jsonl")
    files = glob.glob(search_pattern)

    if not files:
        print(f"❌ 错误: 在目录 '{log_dir}' 下没有找到任何 '*cases.jsonl' 文件。")
        print("   请检查目录路径是否正确。")
        return

    print(f"📂 正在处理目录: {log_dir}")
    print(f"🔍 发现 {len(files)} 个日志文件: {[os.path.basename(f) for f in files]}")
    print("-" * 50)

    # 全局统计变量
    total_samples = 0
    total_em = 0.0
    total_f1 = 0.0

    # 2. 遍历每个文件读取数据
    for file_path in files:
        file_count = 0
        file_em = 0.0
        file_f1 = 0.0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        # 提取 metrics 字段
                        if "metrics" in data:
                            em = data["metrics"].get("em", 0)
                            f1 = data["metrics"].get("f1", 0)
                            
                            # 累加文件级统计
                            file_em += em
                            file_f1 += f1
                            file_count += 1
                    except json.JSONDecodeError:
                        print(f"⚠️ 警告: 文件 {os.path.basename(file_path)} 中有一行 JSON 解析失败，跳过。")
            
            # 输出单文件统计
            if file_count > 0:
                print(f"   📄 {os.path.basename(file_path)}: 样本数={file_count}, Avg EM={file_em/file_count:.4f}")
            
            # 累加到全局
            total_samples += file_count
            total_em += file_em
            total_f1 += file_f1

        except Exception as e:
            print(f"❌ 读取文件 {file_path} 失败: {e}")

    # 3. 计算并输出全局平均值
    print("-" * 50)
    if total_samples == 0:
        print("❌ 没有找到有效的样本数据。")
    else:
        avg_em = total_em / total_samples
        avg_f1 = total_f1 / total_samples

        print(f"✅ 统计完成 (Total Samples: {total_samples})")
        print(f"🏆 Average EM: {avg_em:.4f}  ({avg_em*100:.2f}%)")
        print(f"🏆 Average F1: {avg_f1:.4f}  ({avg_f1*100:.2f}%)")
    print("-" * 50)

if __name__ == "__main__":
    # ================= 配置区域 =================
    # [在此处修改] 你的日志目录路径
    LOG_DIR = "logs/eval_wtq_sft_step128_len256_blocksize32_checkpoint-final"
    # ===========================================
    
    # 自动处理路径结尾的斜杠问题，防止报错
    if not os.path.exists(LOG_DIR):
        print(f"❌ 路径不存在: {LOG_DIR}")
        print("请在代码底部的 '配置区域' 修改 LOG_DIR 变量。")
    else:
        calculate_metrics(LOG_DIR)