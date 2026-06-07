#!/usr/bin/env bash
set -euo pipefail
python src/log_parser.py --log_dir logs --out results/log_summary.md --json results/log_summary.json
python src/compare_methods.py --out results/method_cost_table.md
