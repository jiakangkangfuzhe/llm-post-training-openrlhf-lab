#!/usr/bin/env python3
"""Build tiny local datasets for post-training smoke tests.

These are not benchmark results. They are minimal files to validate data loading,
answer extraction, reward logic, and script wiring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("data/toy"))
    args = parser.parse_args()
    out = args.out_dir

    sft_rows = [
        {"prompt": "Solve: 12 + 7", "response": "12 + 7 = 19. #### 19"},
        {"prompt": "Solve: 3 * 9", "response": "3 times 9 equals 27. #### 27"},
    ]
    dpo_rows = [
        {
            "prompt": "Solve: 12 + 7",
            "chosen": "12 + 7 = 19. #### 19",
            "rejected": "12 + 7 = 20. #### 20",
        },
        {
            "prompt": "Solve: 5 * 6",
            "chosen": "5 times 6 equals 30. #### 30",
            "rejected": "The answer is 11. #### 11",
        },
    ]
    gsm8k_rows = [
        {"question": "Tom has 2 apples and buys 3 more. How many apples?", "answer": "#### 5", "gold_answer": "5"},
        {"question": "A box has 4 bags with 6 candies each. Total candies?", "answer": "#### 24", "gold_answer": "24"},
    ]

    write_jsonl(out / "sft.jsonl", sft_rows)
    write_jsonl(out / "dpo_pairs.jsonl", dpo_rows)
    write_jsonl(out / "gsm8k_rlvr.jsonl", gsm8k_rows)
    print(f"Wrote toy datasets to {out}")


if __name__ == "__main__":
    main()
