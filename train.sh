#!/bin/bash

# ================= 1. 显卡与环境设置 =================
# 既然 0,1 空闲，直接指定
export CUDA_VISIBLE_DEVICES=0,1
# 优化：防止 P2P 通信卡死（虽然 H200 一般没事，加个保险）
export NCCL_P2P_DISABLE=0 
export NCCL_IB_DISABLE=0

NUM_GPUS=2

# ================= 2. 路径配置 =================
MODEL_PATH="/mnt/models/LLaDA-8B-Instruct"
# 输出目录带上参数标识，方便区分
OUTPUT_DIR="models/llada_h200_2gpu_4k_bs128"
DATA_PATH="data/llada_sft_final_train.jsonl"

mkdir -p "$OUTPUT_DIR"

echo "----------------------------------------"
echo "🚀 Starting SFT on H200 x $NUM_GPUS"
echo "Model: $MODEL_PATH"
echo "Strategy: Batch=8/GPU * 2GPUs * 8Accum = 128 Global BS"
echo "Context: 4096 | VRAM: 141GB (Plenty!)"
echo "----------------------------------------"

# ================= 3. 启动命令 =================
# 优化说明：
# H200 显存极大，单卡 Batch=2 太小会导致 GPU 空转。
# 这里改为 Batch=8，Accum=8 -> 总 BS 依然是 128 (2*8*8=128)
# 这样能大幅提升吞吐量 (Samples/s)

accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    --num_processes $NUM_GPUS \
    examples/llada/sft.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_args "$DATA_PATH" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
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
    --truncation "filter" \
    --dataloader_num_workers 8