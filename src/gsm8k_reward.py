#!/usr/bin/env python3
"""Rule-based verifier reward for GSM8K-style answers.

This file is useful for RLVR / GRPO smoke tests. It exposes small pure-Python
functions so it can be imported by training wrappers or used offline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

try:
    from answer_extractor import extract_answer, normalize_number
except ImportError:  # package-style import
    from .answer_extractor import extract_answer, normalize_number


def gsm8k_reward(response: str, gold_answer: str | int | float, correct_reward: float = 1.0, wrong_reward: float = 0.0) -> float:
    pred = extract_answer(response)
    gold = normalize_number(gold_answer)
    return correct_reward if pred is not None and pred == gold else wrong_reward


def batch_reward(responses: Iterable[str], gold_answers: Iterable[str | int | float]) -> list[float]:
    return [gsm8k_reward(r, g) for r, g in zip(responses, gold_answers)]


def evaluate_jsonl(input_path: Path, output_path: Path) -> dict[str, float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    correct = 0
    rewards: list[float] = []
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            response = row.get("response") or row.get("model_response") or ""
            gold = row.get("gold_answer") or row.get("answer")
            pred = extract_answer(response)
            gold_norm = normalize_number(gold)
            reward = gsm8k_reward(response, gold)
            total += 1
            correct += int(reward > 0)
            rewards.append(reward)
            row.update({
                "extracted_answer": pred,
                "gold_normalized": gold_norm,
                "reward": reward,
                "is_correct": bool(reward > 0),
            })
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="JSONL with response/model_response and gold_answer/answer")
    parser.add_argument("--output", type=Path, default=Path("results/gsm8k_reward_eval.jsonl"))
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        assert gsm8k_reward("#### 18", "18") == 1.0
        assert gsm8k_reward("Final answer is 19", "18") == 0.0
        assert gsm8k_reward("Therefore \\boxed{1,200.0}", "1200") == 1.0
        print("gsm8k_reward self-test passed")
        return

    if not args.input:
        raise SystemExit("Provide --input or use --self_test")
    metrics = evaluate_jsonl(args.input, args.output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
