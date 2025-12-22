# 📊 LLaDA-Table-SFT

> **Table Reasoning Fine-Tuning based on LLaDA**

这是一个基于 **dLLM** 框架进行二次开发的表格推理（Table Reasoning）微调项目。本项目主要针对 **LLaDA-8B** 模型进行 SFT (Supervised Fine-Tuning)，通过引入表格数据，使其能够更好地理解表格结构并进行逻辑推理。

🔗 **Upstream Repository:** 本项目基于 [dLLM (Simple Diffusion Language Modeling)](https://www.google.com/search?q=https://github.com/GSAI-ML/LLaDA) 修改。如需查看原始框架的详细文档或底层实现，请访问原仓库。

## 🛠️ 环境准备 (Setup)

### Step 1: 创建并激活新环境

建议使用 Python 3.10 环境：

```Bash
conda create -n dllm python=3.10 -y
conda activate dllm
```

### Step 2: 安装依赖

安装 PyTorch 及项目依赖。

```Bash
pip install -r requirements.txt
```

### Step 3: 安装本项目 (dLLM)

以编辑模式安装，方便调试代码：

```Bash
pip install -e .
```

## 📂 数据准备 (Data Preparation)

本项目重构了数据加载逻辑，支持直接加载本地的 JSONL 数据集，并支持训练集与测试集的划分。

1. **下载与格式化**： 运行下载脚本，将 HuggingFace 数据集转换为 LLaDA 训练所需的格式。

   ```Bash
   python download_data.py
   # 输出: data/table_llada_train.jsonl
   ```

2. **划分数据集**： 运行切分脚本，按照 **90% : 10%** 的比例将数据划分为训练集和测试集。

   ```Bash
   python split_data.py
   # 输出: 
   #   ├── data/table_llada_train_train.jsonl (Training Set)
   #   └── data/table_llada_train_test.jsonl  (Evaluation Set)
   ```

## 🚀 训练示例 (Training)

`examples/llada/sft.py` 已经适配了本地数据加载，并修复了在 DeepSpeed 环境下的日志显示问题。

### 单机多卡 LoRA 微调 (Local w/ 3x4090)

以下命令使用 **DeepSpeed ZeRO-2** 配置，进行 **BF16 + LoRA** 训练。我们显式配置了评估策略 (`eval_steps`)，以便在训练过程中实时观察 Loss 变化。

```Bash
# 1. 设置环境变量 (关闭 wandb 以便专注本地 log)
export WANDB_MODE=disabled

# 2. 启动训练
accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    --num_processes 3 \
    examples/llada/sft.py \
    --model_name_or_path "/path/to/your/LLaDA-8B-Instruct" \
    --output_dir "models/llada_table_lora_run" \
    --dataset_args "data/table_llada_train_train.jsonl" \
    --eval_dataset_args "data/table_llada_train_test.jsonl" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --load_in_4bit True \
    --lora True \
    --bf16 True \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --eval_strategy "steps" \
    --eval_steps 5 \
    --max_length 2048
```

### 关键参数说明

- `--load_in_4bit True`: 启用 4-bit 量化，大幅降低显存占用（适合消费级显卡）。
- `--eval_steps 5`: 每训练 5 步进行一次评估，监控模型是否过拟合。
- `--dataset_args` / `--eval_dataset_args`: 分别指定训练集和测试集路径。
- `--max_length 2048`: 设定最大Token序列长度，超过部分将被丢弃（取决于代码逻辑，通常建议配合数据预处理）。

> **💡 Note:**
>
> - **日志监控**：训练日志（Step Loss）将直接打印在控制台。如果使用了 DeepSpeed 导致控制台不刷新，可以结合 `mdlm.py` 中的修改查看实时 Eval Loss。
> - **集群运行**：如果在 Slurm/FSDP 环境运行，可直接复用 dLLM 原有的 `scripts/train.slurm.sh`，只需将脚本指向本仓库的 `sft.py` 即可。

## 📝 主要修改点 (Key Modifications)

为了适配表格任务及增强调试体验，本项目对原始 dLLM 框架进行了以下修改：

1. **数据加载 (`examples/llada/sft.py`)**:
   - 修改了 Dataset 加载逻辑，支持通过 `--eval_dataset_args` 参数读取本地 JSONL 验证集。
   - 调整了数据预处理流程，确保 Tokenize 和 长度过滤 的顺序正确。
2. **日志增强 (`dllm/core/trainers/mdlm.py`)**:
   - 在底层的 `compute_loss` 函数中增加了强制输出 (`flush=True`)，解决了在 DeepSpeed 多进程环境下进度条遮挡或吞没 Eval Loss 的问题，确保能实时看到评估进度。
3. **辅助脚本**:
   - 新增 `download_data.py` 和 `split_data.py`，用于自动化处理 HuggingFace 表格数据。
4. **配置适配**:
   - `scripts/accelerate_configs/` 下新增/调整了适配 LoRA 的 DeepSpeed 配置文件。