#!/bin/bash

# ================= 1. 显卡与环境设置 =================
export CUDA_VISIBLE_DEVICES=0,1
# H200 通信优化
export NCCL_P2P_DISABLE=0 
export NCCL_IB_DISABLE=0

NUM_GPUS=2

# ================= 2. 路径配置 =================
# [核心修改] 基座模型指向你的 SOTA Checkpoint
# 请确保这个路径下有 config.json, pytorch_model.bin (或safetensors) 等文件
# 如果没有完整文件，可能需要先转换或手动复制 config.json
MODEL_PATH="models/LLaDA-Table-Flash-Verifier/checkpoint-1800"

# [核心修改] 输出目录：新的通用模型
OUTPUT_DIR="models/Table-LLaDA-Generalist"

# [核心修改] 数据路径：混合数据
DATA_PATH="data/llada_sft_mixed_train.jsonl"

mkdir -p "$OUTPUT_DIR"

echo "----------------------------------------"
echo "🚀 Starting Mixed Generalist SFT (Warm-Start)"
echo "Base Model: $MODEL_PATH (TabFact SOTA)"
echo "Data: $DATA_PATH (Mixed TabFact + WTQ)"
echo "Learning Rate: 5e-7 (Fine-tuning mode)"
echo "----------------------------------------"

# ================= 3. 启动命令 =================
accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    --num_processes $NUM_GPUS \
    examples/llada/sft.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_args "$DATA_PATH" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-7 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --load_in_4bit False \
    --lora False \
    --bf16 True \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --max_length 4096 \
    --truncation "filter" \
    --dataloader_num_workers 8