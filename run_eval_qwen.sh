#!/bin/bash

# ================= 配置区域 =================
# 这里填你要用的显卡，比如 0 1 2，或者 1 2 3
GPUS=(0 1 2) 
MODEL_PATH="/home/zjusst/hxy/llada/models/Qwen/Qwen2.5-7B-Instruct" 
DATA_PATH="data/wikitq_test.jsonl"
LOG_DIR="logs/wtq_qwen_eval"
GEN_LENGTH=64
SCRIPT_NAME="evaluate_baseline_qwen.py"
# ===========================================

NUM_SHARDS=${#GPUS[@]}
mkdir -p "$LOG_DIR"

echo "正在清理旧进程..."
pkill -f "$SCRIPT_NAME"
sleep 2

echo "---------------------------------------------------"
echo "Starting Qwen Evaluation on GPUs: ${GPUS[*]}"
echo "---------------------------------------------------"

for ((i=0; i<NUM_SHARDS; i++)); do
    # 获取真实的物理 GPU ID
    GPU_ID=${GPUS[$i]}
    SHARD_ID=$i
    
    echo "Starting Worker $SHARD_ID on Physical GPU $GPU_ID..."
    
    # [关键修正] 必须加 CUDA_VISIBLE_DEVICES！
    # 这样每个 Python 进程都以为自己独占了一张卡 (cuda:0)，实际上它们被隔离到了不同物理卡上
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python $SCRIPT_NAME \
        --gpu_id $GPU_ID \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATA_PATH" \
        --log_dir "$LOG_DIR" \
        --shard_id $SHARD_ID \
        --num_shards $NUM_SHARDS \
        --gen_length $GEN_LENGTH \
        > "$LOG_DIR/nohup_gpu${GPU_ID}.log" 2>&1 &
        
done

wait
echo "All workers finished! Check logs in $LOG_DIR."