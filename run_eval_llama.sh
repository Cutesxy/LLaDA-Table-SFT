#!/bin/bash

# ================= 配置区域 =================
GPUS=(0 1 2) # 你想用的卡

# 修改为你的 Llama 路径
MODEL_PATH="/home/zjusst/hxy/llada/models/Meta-Llama-3.1-8B-Instruct" 
DATA_PATH="data/table_llada_train_test.jsonl"
LOG_DIR="logs/llama_table_eval"

GEN_LENGTH=512
# ===========================================

NUM_SHARDS=${#GPUS[@]}
mkdir -p "$LOG_DIR"

echo "---------------------------------------------------"
echo "Starting Llama Baseline Evaluation on ${NUM_SHARDS} GPUs"
echo "---------------------------------------------------"

for ((i=0; i<NUM_SHARDS; i++)); do
    GPU_ID=${GPUS[$i]}
    SHARD_ID=$i
    
    echo "Starting Worker $SHARD_ID on GPU $GPU_ID..."
    
    # 显存隔离并启动
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate_baseline_llama_vllm.py \
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
echo "All Llama workers finished! Check $LOG_DIR for results."