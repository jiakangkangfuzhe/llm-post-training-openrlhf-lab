#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
DATASET=${DATASET:-data/toy/dpo_pairs.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/dpo_smoke}
DS_CONFIG=${DS_CONFIG:-configs/deepspeed_zero3.json}

mkdir -p "${OUTPUT_DIR}" logs
python src/build_toy_data.py --out_dir data/toy

# Common DPO fields: prompt / chosen / rejected. Check your OpenRLHF version if flags differ.
deepspeed --module openrlhf.cli.train_dpo \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset "${DATASET}" \
  --prompt_key prompt \
  --chosen_key chosen \
  --rejected_key rejected \
  --save_path "${OUTPUT_DIR}" \
  --max_len 1024 \
  --train_batch_size 8 \
  --micro_train_batch_size 1 \
  --learning_rate 5e-6 \
  --beta 0.1 \
  --bf16 \
  --gradient_checkpointing \
  --zero_stage 3 \
  --deepspeed_config "${DS_CONFIG}" \
  2>&1 | tee logs/dpo_smoke.log
