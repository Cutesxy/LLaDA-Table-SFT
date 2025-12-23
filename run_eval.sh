#!/bin/bash

# ================= 配置区域 =================
# 1. 设置你想使用的 GPU ID 列表 (空格分隔)
#    例如想用 0,1,2 卡，就写: GPUS=(0 1 2)
GPUS=(0 1 2)

# 2. 模型和数据路径
MODEL_PATH="models/LLaDA-Table-SFT/checkpoint-final"
DATA_PATH="data/table_llada_train_test.jsonl"
LOG_DIR="logs/llada_table_eval"

# 3. 其他参数
GEN_LENGTH=512
STEPS=128
# ===========================================

# 自动计算总分片数 (就是 GPU 的数量)
NUM_SHARDS=${#GPUS[@]}

echo "---------------------------------------------------"
echo "Starting Parallel Evaluation on ${NUM_SHARDS} GPUs: ${GPUS[*]}"
echo "---------------------------------------------------"

# 循环启动每一个 GPU 任务
for ((i=0; i<NUM_SHARDS; i++)); do
    GPU_ID=${GPUS[$i]}
    SHARD_ID=$i
    
    echo "Starting Worker $SHARD_ID on GPU $GPU_ID..."
    
    # [关键] CUDA_VISIBLE_DEVICES 限制该进程只能看到这一张卡
    # 这样 python 脚本内部看到的总是 "cuda:0"，逻辑就简单了
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    # 后台运行 (&)
    python evaluate_llada_8b.py \
        --gpu_id $GPU_ID \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATA_PATH" \
        --log_dir "$LOG_DIR" \
        --shard_id $SHARD_ID \
        --num_shards $NUM_SHARDS \
        --gen_length $GEN_LENGTH \
        --steps $STEPS \
        > "$LOG_DIR/nohup_gpu${GPU_ID}.log" 2>&1 &
        
done

# 等待所有后台任务结束
echo "---------------------------------------------------"
echo "All workers started! Waiting for them to finish..."
wait
echo "All Done!"

# [可选] 自动合并所有日志算总分
echo "Calculating Final Metrics..."
# 简单的 grep 统计 (你可以之后再写脚本精确统计)
cat $LOG_DIR/eval_gpu*_metrics.log | grep "EM:" > $LOG_DIR/all_results.txt
echo "Results merged to $LOG_DIR/all_results.txt"