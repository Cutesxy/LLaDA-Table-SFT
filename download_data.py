import os
from modelscope.hub.snapshot_download import snapshot_download

def download_files():
    # 配置参数
    dataset_id = 'HanaHxy123/SFT-TableData'
    save_folder = './eval_data'
    
    # 确保文件夹存在
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"已创建文件夹: {save_folder}")
    
    print(f"开始下载数据集仓库文件: {dataset_id}")
    
    try:
        # 使用 snapshot_download 直接下载仓库中的所有文件
        # repo_type='dataset' 指定下载的是数据集
        # local_dir 指定下载到本地的哪个目录
        path = snapshot_download(
            dataset_id, 
            repo_type='dataset', 
            local_dir=save_folder
        )
        
        print(f"下载完成，文件保存在: {path}")
        
    except Exception as e:
        print(f"下载失败: {e}")

if __name__ == "__main__":
    download_files()