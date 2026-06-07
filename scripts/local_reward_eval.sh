#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/toy results
python src/build_toy_data.py --out_dir data/toy
cat > results/sample_responses.jsonl <<'EOF'
{"prompt":"Solve: 12 + 7","response":"12 + 7 = 19. #### 19","gold_answer":"19"}
{"prompt":"Solve: 5 * 6","response":"I think it is 31. #### 31","gold_answer":"30"}
EOF
python src/gsm8k_reward.py --input results/sample_responses.jsonl --output results/sample_reward_scored.jsonl
