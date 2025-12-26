#!/bin/bash

# ================= 配置区域 =================
# 1. 设置 GPU ID (空格分隔)
GPUS=(0 1 2)

# 2. 基础模型路径
MODEL_PATH="/home/zjusst/hxy/llada/models/GSAI-ML/LLaDA-8B-Instruct"

# 3. [新增] LoRA Adapter 路径
#    - 如果要跑 LoRA，填入具体路径 (例如 checkpoint-200)
#    - 如果要跑 Base Model (原版)，请将下方变量留空 (即 ADAPTER_PATH="")
ADAPTER_PATH="./models/llada_table_lora_5e-6_2/checkpoint-600"
# ADAPTER_PATH=""  <-- 如果想跑原版，解开这行注释，注释掉上面那行

# 4. 数据和日志
DATA_PATH="data/table_llada_train_test.jsonl"
LOG_DIR="logs/llada_table_eval_128"

# 5. 生成参数
GEN_LENGTH=512
STEPS=128  # 注意：通常评估时步数不用像训练那么大，128或64通常够了，你之前写的是512步，如果需要可以改回
# ===========================================

# 自动计算总分片数
NUM_SHARDS=${#GPUS[@]}

# 如果跑的是 LoRA，建议把日志文件夹区分开，防止混淆 (可选)
if [ -n "$ADAPTER_PATH" ]; then
    CKPT_NAME=$(basename "$ADAPTER_PATH")
    LOG_DIR="${LOG_DIR}_${CKPT_NAME}"
fi

mkdir -p "$LOG_DIR"

echo "---------------------------------------------------"
echo "Starting Parallel Evaluation on ${NUM_SHARDS} GPUs: ${GPUS[*]}"
echo "Mode: $(if [ -n "$ADAPTER_PATH" ]; then echo "LoRA ($ADAPTER_PATH)"; else echo "Base Model"; fi)"
echo "Logs will be saved to: $LOG_DIR"
echo "---------------------------------------------------"

# 循环启动每一个 GPU 任务
for ((i=0; i<NUM_SHARDS; i++)); do
    GPU_ID=${GPUS[$i]}
    SHARD_ID=$i
    
    echo "Starting Worker $SHARD_ID on GPU $GPU_ID..."
    
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    # 构建基础命令
    CMD="python evaluate_llada_8b.py \
        --gpu_id $GPU_ID \
        --model_path \"$MODEL_PATH\" \
        --dataset_path \"$DATA_PATH\" \
        --log_dir \"$LOG_DIR\" \
        --shard_id $SHARD_ID \
        --num_shards $NUM_SHARDS \
        --gen_length $GEN_LENGTH \
        --steps $STEPS"

    # [关键修改] 如果配置了 Adapter 路径，则追加参数
    if [ -n "$ADAPTER_PATH" ]; then
        CMD="$CMD --adapter_path \"$ADAPTER_PATH\""
    fi
    
    # 后台运行
    eval $CMD > "$LOG_DIR/nohup_gpu${GPU_ID}.log" 2>&1 &
        
done

# 等待所有后台任务结束
echo "---------------------------------------------------"
echo "All workers started! Waiting for them to finish..."
wait
echo "All Done! Check results in $LOG_DIR"