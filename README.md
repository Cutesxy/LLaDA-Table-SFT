# LLaDA-Table-SFT

这是一个基于 **dLLM** 框架进行二次开发的表格推理（Table Reasoning）微调项目。

本项目主要针对 **LLaDA-8B** 模型进行 **SFT (Supervised Fine-Tuning)**，使其能够更好地理解表格结构并进行逻辑推理。

> 🔗 **Upstream Repository**: 本项目基于 [dLLM (Simple Diffusion Language Modeling)](https://github.com/ZHZisZZ/dllm) 修改。如需查看原始框架的详细文档或底层实现，请访问原仓库。

## 环境准备 (Setup)

请确保环境已安装 `torch` 和 `deepspeed`

```bash
# === Step 1: 创建并激活新环境 ===
# 创建一个名为 dllm 的环境，指定 Python 3.10
conda create -n dllm python=3.10 -y

# 激活环境
conda activate dllm

# === Step 2: 安装依赖 ===
# 这一步会自动安装 PyTorch 和上述列表中的所有包
pip install -r requirements.txt

# === Step 3: 安装本项目 (dLLM) ===
pip install -e .
```

## 数据准备 (Data Preparation)

本项目修改了数据加载逻辑，支持直接加载本地的 JSONL 数据集。

**生成训练数据**： 运行根目录下的脚本，将数据集下载并转换为 LLaDA 训练所需的格式。

```bash
python download_data.py
```

运行成功后，数据将生成在 `data/table_llada_train.jsonl`。

## 训练示例 (Training)

`examples/llada/sft.py` 已经适配了本地数据加载。你可以使用 `accelerate` 在单机或集群上启动训练。

### 单机多卡 LoRA 微调示例 (Local w/ 3x4090)

以下命令使用 **DeepSpeed ZeRO-2** 配置，进行 BF16 + LoRA 训练：

```bash
# 1. 准备加速配置 (如 scripts/accelerate_configs/zero2.yaml)
# 2. 设置环境变量 (可选)
export WANDB_MODE=disabled

# 3. 启动训练
accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/llada/sft.py \
    --model_name_or_path "/path/to/your/LLaDA-8B-Instruct" \
    --output_dir "models/llada_table_lora_run" \
    --dataset_args "data/table_llada_train.jsonl" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --load_in_4bit False \
    --lora True \
    --bf16 True \
    --logging_steps 1 \
    --save_strategy "epoch"
```

> **Note**:
>
> - 请根据实际情况修改 `--model_name_or_path`。
> - 如果在集群上运行 (Slurm/FSDP)，可直接复用 dLLM 原有的 `scripts/train.slurm.sh` 或 `fsdp.yaml` 配置，只需指向本仓库的 `sft.py` 即可。

## 主要修改点 (Key Modifications)

为了适配表格任务，主要变动如下：

1. **`examples/llada/sft.py`**: 修改了 `train()` 函数中的 Dataset 加载逻辑，支持读取本地 JSONL 文件。
2. **`download_data.py`**: 新增脚本，用于从 HuggingFace 下载并格式化表格数据。
3. **`scripts/accelerate_configs/`**: 新增/调整了适配 LoRA 的 DeepSpeed 配置文件。