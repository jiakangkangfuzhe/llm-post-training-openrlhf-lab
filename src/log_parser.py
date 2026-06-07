#!/usr/bin/env python3
"""Parse training logs and summarize key post-training signals.

The parser is regex-based and intentionally tolerant. It can catch common fields
from DeepSpeed/OpenRLHF-style logs such as loss, reward, kl, lr, step time, and
CUDA memory strings.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
PATTERNS = {
    "loss": re.compile(rf"(?:loss|train_loss)[:= ]+({FLOAT})", re.I),
    "reward": re.compile(rf"(?:reward|mean_reward)[:= ]+({FLOAT})", re.I),
    "kl": re.compile(rf"(?:kl|kl_div|kl_coef)[:= ]+({FLOAT})", re.I),
    "lr": re.compile(rf"(?:lr|learning_rate)[:= ]+({FLOAT})", re.I),
    "step_time": re.compile(rf"(?:step_time|time/step|sec/step)[:= ]+({FLOAT})", re.I),
    "gpu_mem_gb": re.compile(rf"(?:max_memory|gpu_mem|memory).*?({FLOAT})\s*(?:GB|GiB)", re.I),
}


def parse_file(path: Path) -> dict[str, float | int | str]:
    values: dict[str, list[float]] = {k: [] for k in PATTERNS}
    steps = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "step" in line.lower():
                steps += 1
            for name, pat in PATTERNS.items():
                for m in pat.finditer(line):
                    try:
                        values[name].append(float(m.group(1)))
                    except ValueError:
                        pass
    summary: dict[str, float | int | str] = {"file": str(path), "lines_with_step": steps}
    for name, vals in values.items():
        if vals:
            summary[f"{name}_last"] = vals[-1]
            summary[f"{name}_avg"] = mean(vals)
            summary[f"{name}_n"] = len(vals)
    return summary


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return "No logs found.\n"
    keys = sorted({k for r in rows for k in r.keys()})
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=Path, default=Path("logs"))
    parser.add_argument("--out", type=Path, default=Path("results/log_summary.md"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    logs = sorted(args.log_dir.glob("*.log")) if args.log_dir.exists() else []
    rows = [parse_file(p) for p in logs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_markdown(rows), encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Parsed {len(rows)} logs -> {args.out}")


if __name__ == "__main__":
    main()
