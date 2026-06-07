.PHONY: data test parse clean

data:
	python src/build_toy_data.py --out_dir data/toy

test:
	python src/gsm8k_reward.py --self_test
	python src/compare_methods.py --out results/method_cost_table.md

parse:
	python src/log_parser.py --log_dir logs --out results/log_summary.md

clean:
	rm -rf __pycache__ src/__pycache__ outputs tmp
