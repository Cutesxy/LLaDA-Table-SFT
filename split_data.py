import json
import random
import os

def split_dataset(input_file, train_ratio=0.9, seed=42):
    # 设置随机种子，保证每次切出来的结果一样（可复现）
    random.seed(seed)
    
    print(f"正在读取数据: {input_file} ...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_len = len(lines)
    print(f"原始数据共 {total_len} 条")
    
    # 随机打乱
    random.shuffle(lines)
    
    # 计算切分点
    split_idx = int(total_len * train_ratio)
    
    train_data = lines[:split_idx]
    test_data = lines[split_idx:]
    
    # 构建输出文件名
    dir_name = os.path.dirname(input_file)
    base_name = os.path.basename(input_file)
    name_root, ext = os.path.splitext(base_name)
    
    train_output = os.path.join(dir_name, f"{name_root}_train{ext}")
    test_output = os.path.join(dir_name, f"{name_root}_test{ext}")
    
    # 保存
    with open(train_output, 'w', encoding='utf-8') as f:
        f.writelines(train_data)
        
    with open(test_output, 'w', encoding='utf-8') as f:
        f.writelines(test_data)
        
    print("-" * 30)
    print(f"切分完成！")
    print(f"训练集 (90%): {len(train_data)} 条 -> {train_output}")
    print(f"测试集 (10%): {len(test_data)} 条 -> {test_output}")
    print("-" * 30)

if __name__ == "__main__":
    # 修改这里为你实际的文件路径
    input_path = "data/table_llada_train.jsonl" 
    
    if os.path.exists(input_path):
        split_dataset(input_path)
    else:
        print(f"错误：找不到文件 {input_path}")