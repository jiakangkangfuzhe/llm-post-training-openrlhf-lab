#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
DATASET=${DATASET:-data/toy/sft.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/sft_smoke}
DS_CONFIG=${DS_CONFIG:-configs/deepspeed_zero3.json}

mkdir -p "${OUTPUT_DIR}" logs
python src/build_toy_data.py --out_dir data/toy

# Adjust --input_key/--output_key if your OpenRLHF version expects different names.
deepspeed --module openrlhf.cli.train_sft \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset "${DATASET}" \
  --input_key prompt \
  --output_key response \
  --save_path "${OUTPUT_DIR}" \
  --max_len 1024 \
  --train_batch_size 8 \
  --micro_train_batch_size 1 \
  --learning_rate 5e-6 \
  --bf16 \
  --gradient_checkpointing \
  --zero_stage 3 \
  --deepspeed_config "${DS_CONFIG}" \
  2>&1 | tee logs/sft_smoke.log
