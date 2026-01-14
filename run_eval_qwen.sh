#!/bin/bash

# ================= 配置区域 =================
GPUS=(0 1 2) 
# 确保你的 Qwen 模型路径是正确的
MODEL_PATH="/home/zjusst/hxy/llada/models/Qwen/Qwen2.5-7B-Instruct" 
DATA_PATH="data/wikitql_test.jsonl"
LOG_DIR="logs/wtq_cot_qwen_eval_v1" # 建议更新日志目录名

# [关键参数修改]
GEN_LENGTH=256
SCRIPT_NAME="main_eval.py"   # <--- 1. 改成主程序 main_eval.py
TASK_NAME="wtq-cot"              # <--- 2. 指定任务 (必须匹配 tasks.py)
MODEL_TYPE="hf"              # <--- 3. 指定模型类型 (Qwen 属于 HF 系列)
# ===========================================

NUM_SHARDS=${#GPUS[@]}
mkdir -p "$LOG_DIR"

echo "正在清理旧进程..."
pkill -f "$SCRIPT_NAME"
sleep 2

echo "---------------------------------------------------"
echo "Starting Qwen Evaluation on GPUs: ${GPUS[*]}"
echo "Model Type: $MODEL_TYPE | Task: $TASK_NAME"
echo "---------------------------------------------------"

for ((i=0; i<NUM_SHARDS; i++)); do
    GPU_ID=${GPUS[$i]}
    SHARD_ID=$i
    
    echo "Starting Worker $SHARD_ID on Physical GPU $GPU_ID..."
    
    # [关键修正] 
    # 增加 --model_type, --task 等参数
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
echo "All workers finished! Check logs in $LOG_DIR."