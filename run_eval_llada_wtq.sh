#!/bin/bash

# ================= 配置区域 =================
# [设置] 使用的 GPU 卡号
GPUS=(0 1 2 3)

# [关键修改 1] 模型路径指向你正在训练的混合模型 (Generalist)
# 注意：如果你还在训练中，这里可能需要指向具体的 checkpoint (如 checkpoint-400)
# 训练完成后，通常是 checkpoint-final 或直接是输出目录
# [修正] 请确保这是绝对路径，或者相对于当前执行目录的正确相对路径
MODEL_PATH="/mnt/yss/spg-checkpoint/fact_wtq_sft-rl-checkpoint-2300"
# MODEL_PATH="/mnt/models/LLaDA-8B-Instruct"

# --- LoRA 设置 ---
ADAPTER_PATH=""

# 数据路径
DATA_PATH="data/wikitql_test.jsonl"

# [关键修改 2] 任务名 (确保 main_eval.py 里有对应的处理逻辑)
TASK_NAME="wtq-robust"  

# [关键修改 3] 长度设为 32
# WTQ 答案是不定长的，32 足够覆盖 99% 的情况
GEN_LENGTH=256

# [关键修改 4] 步数设为 32
# 生成任务建议用 32 步以保证拼写质量；如果追求极致速度，之后可以测 Step 1
STEPS=128

BLOCK_SIZE=64

# 日志目录：自动包含步数和长度信息，方便对比
LOG_DIR="logs/eval_wtq_sft_rl_step${STEPS}_len${GEN_LENGTH}_blocksize64_checkpoint-2300" 

SCRIPT_NAME="main_eval.py"
MODEL_TYPE="llada"
# ===========================================

NUM_SHARDS=${#GPUS[@]}

if [ -n "$ADAPTER_PATH" ]; then
    CKPT_NAME=$(basename "$ADAPTER_PATH")
    LOG_DIR="${LOG_DIR}_${CKPT_NAME}"
fi
mkdir -p "$LOG_DIR"

echo "---------------------------------------------------"
echo "🚀 Starting WTQ Evaluation (Mixed Generalist)"
echo "GPUs: ${GPUS[*]}"
echo "Model: $MODEL_PATH"
echo "Strategy: Fixed Canvas (Len=$GEN_LENGTH) + EOS Truncation"
echo "Diffusion Steps: $STEPS"
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
        --steps $STEPS"

    if [ -n "$ADAPTER_PATH" ]; then
        CMD="$CMD --adapter_path $ADAPTER_PATH"
    fi
    
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup $CMD > "$LOG_DIR/nohup_gpu${GPU_ID}.log" 2>&1 &
        
done

wait
echo "All done! Logs in $LOG_DIR"