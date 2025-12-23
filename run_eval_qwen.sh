#!/bin/bash

# ================= 配置区域 =================
# 1. GPU ID 列表
GPUS=(4 5 6)

# 2. 模型和数据路径 (这里换成 Qwen 或 Llama 的路径)
MODEL_PATH="/path/to/your/Qwen2.5-7B-Instruct" 
DATA_PATH="data/table_llada_train_test.jsonl"
LOG_DIR="logs/qwen_table_eval"

# 3. 参数
GEN_LENGTH=512
# ===========================================

NUM_SHARDS=${#GPUS[@]}

# [关键修复] 加上下面这一行，确保文件夹存在
mkdir -p "$LOG_DIR"

echo "---------------------------------------------------"
echo "Starting Baseline Evaluation on ${NUM_SHARDS} GPUs: ${GPUS[*]}"

for ((i=0; i<NUM_SHARDS; i++)); do
    GPU_ID=${GPUS[$i]}
    SHARD_ID=$i
    
    echo "Starting Worker $SHARD_ID on GPU $GPU_ID..."
    
    # 启动进程
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate_baseline_qwen.py \
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
echo "Baseline Evaluation Done!"