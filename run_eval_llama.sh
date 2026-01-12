#!/bin/bash

# ================= Config =================
GPUS=(0 1 2) # 使用的 GPU ID
MODEL_PATH="/home/zjusst/hxy/llada/models/Meta-Llama-3.1-8B-Instruct" 
DATA_PATH="data/tabfact_test.jsonl"
LOG_DIR="logs/tabfact_llama_eval" # 建议换个新目录，避免混淆

# [关键参数]
GEN_LENGTH=256
SCRIPT_NAME="main_eval.py"  # <--- 改成新的主程序
TASK_NAME="tabfact"             # <--- 指定任务类型 (必须与 tasks.py 里的注册名一致)
MODEL_TYPE="hf"             # <--- 指定模型类型 (hf 代表 Llama/Qwen)
# ==========================================

NUM_SHARDS=${#GPUS[@]}
mkdir -p "$LOG_DIR"

echo "Cleaning up old processes..."
pkill -f "$SCRIPT_NAME"
sleep 2

echo "---------------------------------------------------"
echo "Starting Integrated Evaluation on GPUs: ${GPUS[*]}"
echo "Model Type: $MODEL_TYPE | Task: $TASK_NAME"
echo "---------------------------------------------------"

for ((i=0; i<NUM_SHARDS; i++)); do
    GPU_ID=${GPUS[$i]}
    SHARD_ID=$i
    
    echo "Starting Worker $SHARD_ID on Physical GPU $GPU_ID..."
    
    # 构造命令
    # 注意新增了 --model_type 和 --task
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python $SCRIPT_NAME \
        --gpu_id $GPU_ID \
        --model_type $MODEL_TYPE \
        --model_path "$MODEL_PATH" \
        --task "$TASK_NAME" \
        --dataset_path "$DATA_PATH" \
        --log_dir "$LOG_DIR" \
        --shard_id $SHARD_ID \
        --num_shards $NUM_SHARDS \
        --gen_length $GEN_LENGTH \
        > "$LOG_DIR/nohup_gpu${GPU_ID}.log" 2>&1 &
        
done

wait
echo "Done. Results are in $LOG_DIR"