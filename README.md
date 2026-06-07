# llm-post-training-openrlhf-lab

A small, resume-oriented lab repository for learning and reproducing LLM post-training workflows with **OpenRLHF**.

The goal is not to claim production-scale RLHF. The goal is to keep runnable scripts, configs, utilities, and experiment notes around:

- SFT / DPO / GRPO / RLVR training flow
- policy model / reference model / reward model / verifier roles
- GSM8K-style rule-based reward and answer extraction
- rollout, reward scoring, KL penalty, and training-log analysis
- memory / throughput / rollout-cost comparison across post-training methods

> Recommended project wording: “基于 OpenRLHF 搭建 LLM post-training 训练流程，复现 SFT、DPO、GRPO/RLVR smoke run，并整理训练日志、显存、rollout 开销与评测闭环。”

## Repository layout

```text
.
├── configs/                    # DeepSpeed and experiment configs
├── data/                       # Tiny toy datasets for smoke tests
├── docs/                       # Method and engineering notes
├── results/                    # Result templates and generated summaries
├── scripts/                    # One-click shell entry points
└── src/                        # Utility code: reward, extraction, log parsing
```

## Quick start

```bash
conda create -n openrlhf-lab python=3.10 -y
conda activate openrlhf-lab
pip install -r requirements.txt

# Generate small local toy datasets.
python src/build_toy_data.py --out_dir data/toy

# Unit-test the GSM8K answer extractor and reward.
python src/gsm8k_reward.py --self_test

# Parse logs into markdown tables.
python src/log_parser.py --log_dir logs --out results/log_summary.md
```

## OpenRLHF smoke scripts

The `scripts/run_*_smoke.sh` files are intentionally parameterized. Set the model and dataset paths before running.

```bash
export MODEL_NAME_OR_PATH=Qwen/Qwen2.5-0.5B-Instruct
export OUTPUT_DIR=outputs/dpo_smoke
bash scripts/run_dpo_smoke.sh
```

For GRPO / RLVR-style experiments, this repo provides `src/gsm8k_reward.py` and `src/answer_extractor.py` so that you can build a verifier reward for math-answer tasks.

## What to commit after you run

After each real run, commit:

1. the exact script/config used;
2. a short log snippet or generated `results/*.md` summary;
3. memory / step time / reward / accuracy notes;
4. failure cases and extraction errors.

Do **not** commit fake benchmark numbers. Keep `results/*_template.md` until you have real logs.
