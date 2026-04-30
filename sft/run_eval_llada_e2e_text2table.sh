#!/bin/bash
set -euo pipefail

# ================= 配置区域 =================
GPUS=(0 1 2 3)

# Base model
MODEL_PATH="/root/autodl-tmp/llada_runs/LLaDA-Table-Block-103k-notf-v1/checkpoint-final"

# LoRA adapter (可留空)
ADAPTER_PATH=""
# ADAPTER_PATH="/root/autodl-tmp/llada_runs/rl/spg/your_run/checkpoint-2800"

# E2E text-to-table test set (messages+meta jsonl)
DATA_PATH="/home/llada/data/e2e_test_text2table_messages_with_meta.jsonl"

TASK_NAME="e2e-text2table"
TEXT2TABLE_METRIC="E"       # E / c / BS-scaled

# 生成参数
# E2E 输出很短，256 足够覆盖且稳定
GEN_LENGTH=256
STEPS=64
BLOCK_SIZE=32

LOG_ROOT="/root/autodl-tmp/llada_runs/logs"
LOG_DIR="${LOG_ROOT}/eval_e2e_t2t_notf103k_step${STEPS}_len${GEN_LENGTH}_blk${BLOCK_SIZE}_metric_${TEXT2TABLE_METRIC}"

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
echo "Starting E2E Text2Table Eval on GPUs: ${GPUS[*]}"
echo "Model: $MODEL_PATH"
echo "Task: $TASK_NAME | Metric: $TEXT2TABLE_METRIC"
echo "Len: $GEN_LENGTH | Steps: $STEPS | Block: $BLOCK_SIZE"
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
        --block_size $BLOCK_SIZE \
        --text2table_metric $TEXT2TABLE_METRIC"

    if [ -n "$ADAPTER_PATH" ]; then
        CMD="$CMD --adapter_path $ADAPTER_PATH"
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID nohup $CMD > "$LOG_DIR/nohup_gpu${GPU_ID}.log" 2>&1 &
done

wait
echo "All done! Logs in $LOG_DIR"

