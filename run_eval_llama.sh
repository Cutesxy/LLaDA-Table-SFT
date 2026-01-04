#!/bin/bash

# ================= Config =================
GPUS=(0 1 2) # GPUs to use
MODEL_PATH="/home/zjusst/hxy/llada/models/Meta-Llama-3.1-8B-Instruct" # Change this!
DATA_PATH="data/wikitq_test.jsonl"
LOG_DIR="logs/wtq_llama_eval"
GEN_LENGTH=64
SCRIPT_NAME="evaluate_baseline_llama.py"
# ==========================================

NUM_SHARDS=${#GPUS[@]}
mkdir -p "$LOG_DIR"

echo "Cleaning up old processes..."
pkill -f "$SCRIPT_NAME"
sleep 2

echo "---------------------------------------------------"
echo "Starting Llama Evaluation on GPUs: ${GPUS[*]}"
echo "---------------------------------------------------"

for ((i=0; i<NUM_SHARDS; i++)); do
    GPU_ID=${GPUS[$i]}
    SHARD_ID=$i
    
    echo "Starting Worker $SHARD_ID on Physical GPU $GPU_ID..."
    
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
echo "Done."