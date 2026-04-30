#!/bin/bash
set -euo pipefail

# ================= 1. 显卡与环境设置 =================
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# 显存碎片优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 日志级别
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
OUTPUT_DIR="/root/autodl-tmp/llada_runs/LLaDA-Table-Block-105k-tf-fast-v1"
DATA_PATH="/home/llada/data/llada_sft_train_all_105357_messages_mix40k_tablegptaug.jsonl"
LOG_DIR="/root/autodl-tmp/llada_runs/logs"

# ================= 3. TF 关键参数 =================
TF_MAX_STAGES=2
TF_CONFIDENCE_THRESHOLD=0.70
TF_ONLY_MASKED_TOKENS=false

# ================= 4. 训练参数 =================
NUM_EPOCHS=1
MAX_LENGTH=3072
BLOCK_SIZE=32
PER_DEVICE_BS=1
GRAD_ACCUM=16
LR=1e-6

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"

echo "----------------------------------------"
echo "Starting Block Diffusion SFT + Teacher Forcing on $NUM_GPUS GPUs"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Data: $DATA_PATH"
echo "Block Size: $BLOCK_SIZE"
echo "TF: stages=$TF_MAX_STAGES, threshold=$TF_CONFIDENCE_THRESHOLD, only_masked=$TF_ONLY_MASKED_TOKENS"
echo "Epochs: $NUM_EPOCHS"
echo "Batch: per_device=$PER_DEVICE_BS, grad_accum=$GRAD_ACCUM, global=$((PER_DEVICE_BS * NUM_GPUS * GRAD_ACCUM))"
echo "Max Length: $MAX_LENGTH"
echo "ZeRO: stage3"
echo "Save: only checkpoint-final"
echo "----------------------------------------"

ACCELERATE_BIN="/root/miniconda3/envs/dllm/bin/accelerate"

"$ACCELERATE_BIN" launch \
    --config_file scripts/accelerate_configs/zero3.yaml \
    --num_processes $NUM_GPUS \
    examples/llada/block_sft_teacher_forcing.py \
    --model_name_or_path "$MODEL_PATH" \
    --attn_implementation "eager" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_args "$DATA_PATH" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05 \
    --load_in_4bit False \
    --lora False \
    --bf16 True \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --max_length "$MAX_LENGTH" \
    --block_size "$BLOCK_SIZE" \
    --tf_max_stages "$TF_MAX_STAGES" \
    --tf_confidence_threshold "$TF_CONFIDENCE_THRESHOLD" \
    --tf_only_masked_tokens "$TF_ONLY_MASKED_TOKENS" \
    --truncation "right" \
    --dataloader_num_workers 8
