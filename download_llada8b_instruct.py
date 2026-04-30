import os

# 核心配置：设置 Hugging Face 国内镜像源
# 必须在导入 huggingface_hub 之前或最开始设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
SAVE_DIR = "/home/llada/models/LLaDA-8B-Instruct"

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 如果模型是受保护的 (Gated Model)，需要填入你的 HF_TOKEN
    # 也可以在终端中通过 export HF_TOKEN="你的token" 来设置
    token = os.environ.get("HF_TOKEN", None)

    print(f"Start downloading: {MODEL_ID}")
    print(f"Save to: {SAVE_DIR}")
    print(f"Using endpoint: {os.environ.get('HF_ENDPOINT')}")

    local_path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=SAVE_DIR,
        local_dir_use_symlinks=False,   # 直接保存真实文件，更直观
        # resume_download=True,         # 新版本已默认开启断点续传，可注释掉以避免警告
        token=token,
        max_workers=8                   # 可选：增加并发下载数，提升多文件下载速度
    )

    print("Download finished.")
    print(f"Local path: {local_path}")

if __name__ == "__main__":
    main()