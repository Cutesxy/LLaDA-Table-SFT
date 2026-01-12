import os
import json
from modelscope.msdatasets import MsDataset

# 1. 设置保存路径
save_dir = "data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

output_train = os.path.join(save_dir, "wikitq_train.jsonl")
output_test = os.path.join(save_dir, "wikitq_test.jsonl")

dataset_id = 'HanaHxy123/wikitq'

# --- 核心修改：显式指定文件映射 ---
# 告诉 SDK：train 只有那个 7MB 的文件，test 只有那个 800KB 的文件
custom_data_files = {
    'train': 'wikitq_test_train.jsonl', 
    'test':  'wikitq_test_test.jsonl'
}

def save_split(dataset_id, split_name, output_file, data_files_config):
    print(f"--- 正在加载并处理 {split_name} 集 ---")
    
    # 使用 data_files 参数精准定位文件
    ds = MsDataset.load(
        dataset_id, 
        split=split_name, 
        data_files=data_files_config
    )
    
    print(f"加载完成，正在写入: {output_file} ...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in ds:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"{split_name} 集完成！数据量: {len(ds)}\n")

if __name__ == "__main__":
    print(f"开始处理数据集: {dataset_id} (指定文件模式)")
    
    # 处理 Train
    save_split(dataset_id, 'train', output_train, custom_data_files)
    
    # 处理 Test
    save_split(dataset_id, 'test', output_test, custom_data_files)
    
    print("所有任务结束。")