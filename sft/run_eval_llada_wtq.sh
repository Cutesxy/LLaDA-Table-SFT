#!/bin/bash

# ================= 配置区域 =================
# [设置] 使用的 GPU 卡号
GPUS=(0 1 2 3)

# 模型路径
MODEL_PATH="/root/autodl-tmp/llada_runs/LLaDA-Table-Block-103k-notf-v1/checkpoint-final"
# MODEL_PATH="/mnt/models/LLaDA-8B-Instruct"

# --- LoRA 设置 ---
ADAPTER_PATH=""

# 数据路径
DATA_PATH="/home/llada/data/wikitql_test.jsonl"

# 任务名
TASK_NAME="wtq-robust"

# 生成长度与步数
GEN_LENGTH=256
STEPS=128
BLOCK_SIZE=32

# [修改] 日志根目录改到数据盘
LOG_ROOT="/root/autodl-tmp/llada_runs/logs"
LOG_DIR="${LOG_ROOT}/eval_wtq_notf103k_step${STEPS}_len${GEN_LENGTH}_blocksize${BLOCK_SIZE}"

SCRIPT_NAME="main_eval.py"
MODEL_TYPE="llada"
# ===========================================

NUM_SHARDS=${#GPUS[@]}

if [ -n "$ADAPTER_PATH" ]; then
    CKPT_NAME=$(basename "$ADAPTER_PATH")
    LOG_DIR="${LOG_DIR}_${CKPT_NAME}"
fi

mkdir -p "$LOG_ROOT" "$LOG_DIR"

echo "---------------------------------------------------"
echo "🚀 Starting WTQ Evaluation (Mixed Generalist)"
echo "GPUs: ${GPUS[*]}"
echo "Model: $MODEL_PATH"
echo "Strategy: Fixed Canvas (Len=$GEN_LENGTH) + EOS Truncation"
echo "Diffusion Steps: $STEPS"
echo "Block Size: $BLOCK_SIZE"
echo "Logs: $LOG_DIR"
echo "---------------------------------------------------"

for ((i=0; i<NUM_SHARDS; i++)); do
    GPU_ID=${GPUS[$i]}
    SHARD_ID=$i

    echo "Starting Worker $SHARD_ID on GPU $GPU_ID..."

    CMD="python $SCRIPT_NAME \
        --gpu_id $GPU_ID \
        --model_type $MODEL_TYPE \
        --model_path $MODEL_PATH \
        --task $TASK_NAME \
        --dataset_path $DATA_PATH \
        --log_dir $LOG_DIR \
        --shard_id $SHARD_ID \
        --num_shards $NUM_SHARDS \
        --gen_length $GEN_LENGTH \
        --steps $STEPS \
        --block_size $BLOCK_SIZE"

    if [ -n "$ADAPTER_PATH" ]; then
        CMD="$CMD --adapter_path $ADAPTER_PATH"
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID nohup $CMD > "$LOG_DIR/nohup_gpu${GPU_ID}.log" 2>&1 &
done

wait
echo "All done! Logs in $LOG_DIR"
