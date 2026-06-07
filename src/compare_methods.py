#!/usr/bin/env python3
"""Create a qualitative method-cost table for post-training workflows.

This table is deliberately descriptive. Replace notes with measured numbers after
running real SFT/DPO/GRPO/PPO jobs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROWS = [
    ["SFT", "policy", "low", "no", "stable supervised baseline"],
    ["DPO", "policy + frozen reference", "medium", "no online rollout", "pairwise preference optimization"],
    ["PPO/RLHF", "policy + reference + reward + critic/value", "high", "yes", "rollout and reward scoring dominate cost"],
    ["GRPO/RLVR", "policy + reference + verifier/reward", "high", "yes", "group sampling and verifier reward for reasoning tasks"],
]


def to_markdown(rows: list[list[str]]) -> str:
    header = ["method", "models loaded", "relative memory", "rollout", "engineering note"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/method_cost_table.md"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_markdown(ROWS), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
