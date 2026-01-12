#!/bin/bash

# ================= 配置区域 =================
GPUS=(0 1 2)
MODEL_PATH="/home/zjusst/hxy/llada/models/GSAI-ML/LLaDA-8B-Instruct"

# --- LoRA 设置 ---
# 如果跑原版 Base，保持为空字符串: ADAPTER_PATH=""
# 如果跑 LoRA，填入路径，例如: ADAPTER_PATH="./models/llada_table_lora/checkpoint-400"
ADAPTER_PATH=""

DATA_PATH="data/wikitql.jsonl"
LOG_DIR="logs/wtq_llada_eval" # 建议更新日志目录名

# [关键参数]
GEN_LENGTH=64
STEPS=128      # LLaDA 专用步数

# [修改点 1] 指向新的主程序
SCRIPT_NAME="main_eval.py"
# [修改点 2] 指定任务和模型类型
TASK_NAME="wtq"
MODEL_TYPE="llada"
# ===========================================

NUM_SHARDS=${#GPUS[@]}

# 如果跑 LoRA，自动修改日志目录名，方便区分
if [ -n "$ADAPTER_PATH" ]; then
    CKPT_NAME=$(basename "$ADAPTER_PATH")
    LOG_DIR="${LOG_DIR}_${CKPT_NAME}"
fi
mkdir -p "$LOG_DIR"

echo "Cleaning up old processes..."
pkill -f "$SCRIPT_NAME"
sleep 2

echo "---------------------------------------------------"
echo "Starting LLaDA Evaluation on GPUs: ${GPUS[*]}"
echo "Model Type: $MODEL_TYPE | Task: $TASK_NAME"
echo "Mode: $(if [ -n "$ADAPTER_PATH" ]; then echo "LoRA ($ADAPTER_PATH)"; else echo "Base Model"; fi)"
echo "---------------------------------------------------"

for ((i=0; i<NUM_SHARDS; i++)); do
    GPU_ID=${GPUS[$i]}
    SHARD_ID=$i
    
    echo "Starting Worker $SHARD_ID on Physical GPU $GPU_ID..."
    
    # [修改点 3] 构建命令
    # 注意：这里新增了 --model_type, --task
    # 保留了 --steps (因为是 llada)
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
        --steps $STEPS"

    # 如果配置了 LoRA 路径，追加参数
    if [ -n "$ADAPTER_PATH" ]; then
        CMD="$CMD --adapter_path $ADAPTER_PATH"
    fi
    
    # 运行
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup $CMD > "$LOG_DIR/nohup_gpu${GPU_ID}.log" 2>&1 &
        
done

wait
echo "All workers finished! Check results in $LOG_DIR"