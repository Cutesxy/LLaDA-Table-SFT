#!/bin/bash

# ================= 1. 显卡与环境设置 =================
export CUDA_VISIBLE_DEVICES=2,3
# H200 通信优化 (保持您的设置)
export NCCL_P2P_DISABLE=0 
export NCCL_IB_DISABLE=0

NUM_GPUS=2

# ================= 2. 路径配置 =================
# 您的模型路径
MODEL_PATH="/mnt/models/LLaDA-8B-Instruct"

# [修改] 输出目录：Block Diffusion 专用
OUTPUT_DIR="models/LLaDA-Table-Block"

# [保持] 数据路径：数据内容不变，训练方式变了
DATA_PATH="data/llada_sft_final_train.jsonl"

mkdir -p "$OUTPUT_DIR"

echo "----------------------------------------"
echo "🚀 Starting Block Diffusion SFT on H200 x $NUM_GPUS"
echo "Model: $MODEL_PATH"
echo "Block Size: 32"
echo "Strategy: Batch=16/GPU * 2GPUs * 4Accum = 128 Global BS"
echo "Note: Effective Seq Len is 2x due to concat (x_t + x_0)"
echo "----------------------------------------"

# ================= 3. 启动命令 =================
# 注意：Block Diffusion 会在内部把长度翻倍 (Noise + Clean)，
# H200 显存很大应该没问题，如果 OOM 请减小 per_device_train_batch_size

accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    --num_processes $NUM_GPUS \
    examples/llada/block_sft.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_args "$DATA_PATH" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05 \
    --load_in_4bit False \
    --lora False \
    --bf16 True \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --max_length 4096 \
    --block_size 32 \
    --truncation "filter" \
    --dataloader_num_workers 8