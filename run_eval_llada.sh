#!/bin/bash

# ================= 配置区域 =================
GPUS=(0)
MODEL_PATH="/home/zjusst/hxy/llada/models/GSAI-ML/LLaDA-8B-Instruct"

# --- LoRA 设置 ---
# 如果跑原版 Base，保持为空字符串: ADAPTER_PATH=""
# 如果跑 LoRA，填入路径: ADAPTER_PATH="./models/llada_table_lora/checkpoint-400"
ADAPTER_PATH=""

DATA_PATH="data/table_llada_train_test.jsonl"
LOG_DIR="logs/llada_eval"

# [关键参数]
GEN_LENGTH=512  
STEPS=128      # 步数 128 保证生成质量

# [注意] 这里的文件名要和你保存的 Python 脚本一致
SCRIPT_NAME="evaluate_llada_8b_v1.py"
# ===========================================

NUM_SHARDS=${#GPUS[@]}

# 如果跑 LoRA，区分日志目录
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
echo "Mode: $(if [ -n "$ADAPTER_PATH" ]; then echo "LoRA ($ADAPTER_PATH)"; else echo "Base Model"; fi)"
echo "Gen Length: $GEN_LENGTH | Steps: $STEPS"
echo "---------------------------------------------------"

for ((i=0; i<NUM_SHARDS; i++)); do
    GPU_ID=${GPUS[$i]}
    SHARD_ID=$i
    
    echo "Starting Worker $SHARD_ID on Physical GPU $GPU_ID..."
    
    # =======================================================
    # [修复] 这里的变量不要加 \" 转义引号
    # 直接使用变量 $MODEL_PATH 即可，Bash 会自动展开
    # =======================================================
    CMD="python $SCRIPT_NAME \
        --gpu_id $GPU_ID \
        --model_path $MODEL_PATH \
        --dataset_path $DATA_PATH \
        --log_dir $LOG_DIR \
        --shard_id $SHARD_ID \
        --num_shards $NUM_SHARDS \
        --gen_length $GEN_LENGTH \
        --steps $STEPS"

    if [ -n "$ADAPTER_PATH" ]; then
        # [修复] 同理，这里也不要加引号
        CMD="$CMD --adapter_path $ADAPTER_PATH"
    fi
    
    # 使用 CUDA_VISIBLE_DEVICES 隔离显卡
    # $CMD 不需要 eval，直接运行即可
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup $CMD > "$LOG_DIR/nohup_gpu${GPU_ID}.log" 2>&1 &
        
done

wait
echo "All workers finished! Check results in $LOG_DIR"