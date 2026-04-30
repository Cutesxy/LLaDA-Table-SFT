#!/bin/bash

# Generate a random port number between 10000 and 65535
RANDOM_PORT=$((RANDOM % 55536 + 10000))
echo "Using random main_process_port: $RANDOM_PORT"

SAVE_DIR=fsx-checkpoints

DATASET="tablegpt"
# DATASET="gsm8k"
RUN_NAME=${DATASET}_base_spg_mix_beta1.5_weight0.5

MODEL_PATH=/home/william/yss/LLaDA-8B-Instruct
NUM_ITER=4

accelerate launch \
    --config_file ./accelerate.yaml \
    --main_process_port $RANDOM_PORT ./diffu_grpo_train.py \
    --config ./train_tablegpt.yaml \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir ${SAVE_DIR}/spg/$RUN_NAME \
    --trainer spg \
    --forward_type block_random \
    --num_t 2 \
    --min_t 0 \
    --max_t 1 \
    --num_generations 8 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --beta 0.0 \
    --logp_estimation mix \
    --mix_weight 0.5 \
    --eubo_beta 1.5 \