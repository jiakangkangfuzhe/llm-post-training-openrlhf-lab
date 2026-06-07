#!/usr/bin/env python3
"""Check whether OpenRLHF command modules are importable in this environment."""
from __future__ import annotations

import importlib.util

MODULES = [
    "openrlhf.cli.train_sft",
    "openrlhf.cli.train_dpo",
    "openrlhf.cli.train_ppo_ray",
    "openrlhf.cli.train_grpo",
]


def main() -> None:
    ok = True
    for m in MODULES:
        found = importlib.util.find_spec(m) is not None
        print(f"{m}: {'OK' if found else 'NOT FOUND'}")
        ok = ok and found
    if not ok:
        print("Some command modules were not found. Check your OpenRLHF version and adjust scripts accordingly.")


if __name__ == "__main__":
    main()
