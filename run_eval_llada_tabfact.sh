#!/bin/bash

# ================= 配置区域 =================
GPUS=(0 1 2 3)  # 使用的 GPU 编号

# 模型路径（与上面 WTQ 脚本对齐）
MODEL_PATH="/home/llada/models/LLaDA-8B-Instruct"
# MODEL_PATH="/mnt/models/LLaDA-8B-Instruct"

# --- LoRA 设置 ---
ADAPTER_PATH=""
# ADAPTER_PATH="/root/autodl-tmp/llada_runs/your_lora/checkpoint-final"

# 数据路径（与上面脚本风格一致，绝对路径）
DATA_PATH="/home/llada/data/tabfact_test.jsonl"

TASK_NAME="tabfact-robust"
GEN_LENGTH=256
STEPS=128
BLOCK_SIZE=32

# 日志输出到数据盘（与上面 WTQ 脚本对齐）
LOG_ROOT="/root/autodl-tmp/llada_runs/logs"
LOG_DIR="${LOG_ROOT}/eval_tabfact_sft_step${STEPS}_len${GEN_LENGTH}_blocksize${BLOCK_SIZE}_checkpoint-base_model"

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
echo "Starting TabFact Eval on GPUs: ${GPUS[*]}"
echo "Model: $MODEL_PATH"
echo "Task: $TASK_NAME | Len: $GEN_LENGTH | Steps: $STEPS | Block: $BLOCK_SIZE"
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
