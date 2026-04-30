#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

RANDOM_PORT=$((RANDOM % 55536 + 10000))
echo "Using random main_process_port: $RANDOM_PORT"

SAVE_DIR="/root/autodl-tmp/llada_runs/rl"
LOG_DIR="/root/autodl-tmp/llada_runs/logs"
mkdir -p "$LOG_DIR"

export SWANLAB_MODE="disabled"
export SWANLAB_EXP_NAME="llada-spg-sft-rl-exp"
export SWANLAB_PROJ_NAME="llada-spg-sft-rl"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET="table_fact_wtq"
RUN_NAME="${DATASET}_sftfinal_spg_mix_beta1.5_weight0.5_step64_block32_ng2_pbs2_mp1024"
MODEL_PATH="/root/autodl-tmp/llada_runs/LLaDA-Table-Block/checkpoint-final"
NUM_ITER=4

mkdir -p "${SAVE_DIR}/spg/${RUN_NAME}"

accelerate launch \
  --config_file ./accelerate.yaml \
  --num_processes "$NUM_GPUS" \
  --main_process_port "$RANDOM_PORT" \
  ./diffu_grpo_train.py \
  --config ./train_tablegpt.yaml \
  --model_path "$MODEL_PATH" \
  --num_iterations "$NUM_ITER" \
  --dataset "$DATASET" \
  --run_name "$RUN_NAME" \
  --output_dir "${SAVE_DIR}/spg/${RUN_NAME}" \
  --trainer spg \
  --forward_type block_random \
  --num_t 2 \
  --min_t 0 \
  --max_t 1 \
  --num_generations 2 \
  --generation_batch_size 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --max_prompt_length 1024 \
  --max_completion_length 256 \
  --diffusion_steps 64 \
  --block_length 32 \
  --beta 0.0 \
  --logp_estimation mix \
  --mix_weight 0.5 \
  --eubo_beta 1.5
