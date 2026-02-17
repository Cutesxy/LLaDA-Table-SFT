#!/bin/bash

# ================= 配置 =================
TARGET_SCRIPT="./run_flash_verifier.sh"
GPU_IDS="0,1"
CHECK_INTERVAL=30
# [新增] 训练日志存放位置
TRAIN_LOG="train_flash.log"
# =======================================

echo "🐶 看门狗已启动！正在监控 GPU $GPU_IDS ..."
echo "🎯 目标：当 GPU 0/1 上没有 'C' (Compute) 类进程时，启动 $TARGET_SCRIPT"

while true; do
    BUSY_COUNT=$(nvidia-smi -i $GPU_IDS | grep " C " | wc -l)
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

    if [ "$BUSY_COUNT" -eq "0" ]; then
        echo "[$TIMESTAMP] ✅ 检测到 GPU $GPU_IDS 空闲！"
        echo "🚀 正在启动训练任务 (后台运行)..."
        echo "📝 训练日志将写入: $TRAIN_LOG"
        
        chmod +x "$TARGET_SCRIPT"
        
        # =======================================================
        # [关键修改] 使用 nohup 后台启动，并独立定向日志
        # =======================================================
        nohup ./"$TARGET_SCRIPT" > "$TRAIN_LOG" 2>&1 &
        
        # 获取刚才启动的训练进程 PID，方便你确认
        TRAIN_PID=$!
        echo "✅ 训练已启动！PID: $TRAIN_PID"
        
        # 任务启动后，看门狗功成身退
        break
    else
        echo "[$TIMESTAMP] ⏳ GPU $GPU_IDS 忙碌中... (发现 $BUSY_COUNT 个计算进程)"
        # 简化日志，只打印一行
        nvidia-smi -i $GPU_IDS --query-compute-apps=process_name,pid --format=csv,noheader | tr '\n' ' '
        echo "" 
        sleep $CHECK_INTERVAL
    fi
done