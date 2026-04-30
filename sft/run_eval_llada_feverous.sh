#!/bin/bash

# ================= 配置区域 =================
GPUS=(0 1)  # 使用的 GPU 编号
MODEL_PATH="/mnt/hxy/LLaDA-Table-SFT/models/Table-LLaDA-Generalist/checkpoint-1000"
# MODEL_PATH="/mnt/models/LLaDA-8B-Instruct"

# --- LoRA 设置 ---
# 跑 SFT 权重填这里，跑 Base 留空
ADAPTER_PATH="" 
# ADAPTER_PATH="./models/llada_table_lora_1e-4_run/checkpoint-final"

DATA_PATH="data/feverous_test.jsonl"

# [关键修改]
TASK_NAME="feverous"  # <--- 改为新的填空任务
GEN_LENGTH=3                    # <--- 长度设为 3 (2单词 + 1 EOS)
STEPS=1                    # <--- 64步足够生成短词，且更快
BLOCK_SIZE=32
LOG_DIR="logs/feverous_generalist_step${STEPS}_blk${BLOCK_SIZE}_ckpt_1000" 

# 主程序
SCRIPT_NAME="main_eval.py"
MODEL_TYPE="llada"
# ===========================================

NUM_SHARDS=${#GPUS[@]}

if [ -n "$ADAPTER_PATH" ]; then
    CKPT_NAME=$(basename "$ADAPTER_PATH")
    LOG_DIR="${LOG_DIR}_${CKPT_NAME}"
fi
mkdir -p "$LOG_DIR"

# echo "Cleaning up..."
# pkill -f "$SCRIPT_NAME"
# sleep 2

echo "---------------------------------------------------"
echo "Starting TabFact Fixed-Length Eval on GPUs: ${GPUS[*]}"
echo "Task: $TASK_NAME | Mask Length: $GEN_LENGTH | Block Size: $BLOCK_SIZE"
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
        --block_size $BLOCK_SIZE"

    if [ -n "$ADAPTER_PATH" ]; then
        CMD="$CMD --adapter_path $ADAPTER_PATH"
    fi
    
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup $CMD > "$LOG_DIR/nohup_gpu${GPU_ID}.log" 2>&1 &
        
done

wait
echo "All done! Logs in $LOG_DIR"
