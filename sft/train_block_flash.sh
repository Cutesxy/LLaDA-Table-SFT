#!/bin/bash
set -euo pipefail

# ================= 1. 显卡与环境设置 =================
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_DISTRIBUTED_DEBUG=INFO
export ACCELERATE_LOG_LEVEL=info
export TOKENIZERS_PARALLELISM=false

# 缓存放数据盘
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/root/autodl-tmp/.cache/huggingface/datasets
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/.cache/torch_extensions
export TRITON_CACHE_DIR=/root/autodl-tmp/.cache/triton
export TMPDIR=/root/autodl-tmp/.cache/tmp

NUM_GPUS=4

# ================= 2. 路径配置 =================
MODEL_PATH="/home/llada/models/LLaDA-8B-Instruct"
OUTPUT_DIR="/root/autodl-tmp/llada_runs/LLaDA-Table-Block-103k-notf-v1"
DATA_PATH="/home/llada/data/llada_sft_train_all_103587_messages_mix40k_tablegptaug_dianswerable.jsonl"
LOG_DIR="/root/autodl-tmp/llada_runs/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"

echo "----------------------------------------"
echo "Starting Block Diffusion SFT (no-TF) on $NUM_GPUS GPUs"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Data: $DATA_PATH"
echo "Block Size: 32"
echo "Strategy: Batch=1/GPU * 4GPUs * 32Accum = 128 Global BS"
echo "ZeRO: stage3"
echo "Save: checkpoint-final only"
echo "----------------------------------------"

ACCELERATE_BIN="/root/miniconda3/envs/dllm/bin/accelerate"

"$ACCELERATE_BIN" launch \
    --config_file scripts/accelerate_configs/zero3.yaml \
    --num_processes $NUM_GPUS \
    examples/llada/block_sft.py \
    --model_name_or_path "$MODEL_PATH" \
    --attn_implementation "eager" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_args "$DATA_PATH" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05 \
    --load_in_4bit False \
    --lora False \
    --bf16 True \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --max_length 4096 \
    --block_size 32 \
    --truncation "right" \
    --dataloader_num_workers 8
