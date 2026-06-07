#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
DATASET=${DATASET:-data/toy/gsm8k_rlvr.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/grpo_gsm8k_smoke}

mkdir -p "${OUTPUT_DIR}" logs
python src/build_toy_data.py --out_dir data/toy
python src/gsm8k_reward.py --self_test

# OpenRLHF GRPO flags may change across versions. Treat this as a reproducible template.
# If your installed OpenRLHF does not expose train_grpo, run:
#   python src/validate_openrlhf_cli.py
# and update this script according to the current docs.
deepspeed --module openrlhf.cli.train_grpo \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset "${DATASET}" \
  --input_key question \
  --answer_key gold_answer \
  --save_path "${OUTPUT_DIR}" \
  --max_prompt_length 512 \
  --max_completion_length 256 \
  --num_generations 4 \
  --train_batch_size 8 \
  --micro_train_batch_size 1 \
  --learning_rate 1e-6 \
  --bf16 \
  --gradient_checkpointing \
  --zero_stage 3 \
  2>&1 | tee logs/grpo_gsm8k_smoke.log
