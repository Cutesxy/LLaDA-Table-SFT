#!/bin/bash

# ================= 配置区域 =================
GPUS=(0 1 2) # 你要用的显卡ID
MODEL_PATH="/home/zjusst/hxy/llada/models/Qwen/Qwen2.5-7B-Instruct" 
DATA_PATH="data/table_llada_train_test.jsonl"
LOG_DIR="logs/qwen_table_eval"
GEN_LENGTH=512
# ===========================================

NUM_SHARDS=${#GPUS[@]}
mkdir -p "$LOG_DIR"

echo "---------------------------------------------------"
echo "Starting Qwen Baseline Evaluation on ${NUM_SHARDS} GPUs"
echo "---------------------------------------------------"

for ((i=0; i<NUM_SHARDS; i++)); do
    GPU_ID=${GPUS[$i]}
    SHARD_ID=$i
    
    echo "Starting Worker $SHARD_ID on GPU $GPU_ID..."
    
    # [关键修改] 这里不要加 CUDA_VISIBLE_DEVICES 前缀
    # 让 Python 脚本自己去处理显卡绑定
    nohup python evaluate_baseline_qwen_vllm.py \
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
echo "All Qwen workers finished! Check $LOG_DIR for results."