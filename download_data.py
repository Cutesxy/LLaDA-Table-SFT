import os
import json
from modelscope.msdatasets import MsDataset

# 1. 设置保存路径 (在当前目录下创建 data 文件夹)
save_dir = "data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
output_file = os.path.join(save_dir, "table_llada_train.jsonl")

print(f"开始下载数据集: spursgozmy/TableLLaDA_train_data ...")

# 2. 加载数据集 (根据提供的信息)
# 注意：ModelScope 的数据通常会被缓存到 ~/.cache/modelscope 下
ds = MsDataset.load('spursgozmy/TableLLaDA_train_data', subset_name='default', split='train')

print(f"下载完成，正在转换为 JSONL 格式并保存到: {output_file} ...")

# 3. 转换为 JSONL 格式保存
# 这样 dLLM 的 dataset loader 比较容易读取
with open(output_file, 'w', encoding='utf-8') as f:
    for item in ds:
        # item 是一个字典，直接写入一行 JSON
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"处理完成！共有 {len(ds)} 条数据。")