#!/usr/bin/env python3
"""Answer extraction helpers for GSM8K-style math responses.

The functions here are intentionally dependency-light and can be used both in
training-time reward functions and offline evaluation scripts.
"""
from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_HASH_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def normalize_number(text: str | int | float | Decimal | None) -> Optional[str]:
    """Normalize a numeric answer string for exact comparison.

    Examples:
        "1,200.0" -> "1200"
        "$-3.50" -> "-3.5"
    """
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    raw = raw.replace(",", "")
    raw = raw.replace("$", "").replace("%", "").strip()
    # Keep only the first numeric span if the caller passed extra words.
    m = _NUMBER_RE.search(raw)
    if not m:
        return None
    raw = m.group(0).replace(",", "")
    try:
        d = Decimal(raw)
    except InvalidOperation:
        return raw
    if d == d.to_integral_value():
        return str(int(d))
    return format(d.normalize(), "f").rstrip("0").rstrip(".")


def extract_answer(text: str) -> Optional[str]:
    """Extract the final numeric answer from a model response.

    Priority:
    1. GSM8K canonical "#### answer"
    2. LaTeX boxed answer
    3. final line after markers like "answer is"
    4. last number in the full response
    """
    if text is None:
        return None
    text = str(text)

    hash_match = list(_HASH_RE.finditer(text))
    if hash_match:
        return normalize_number(hash_match[-1].group(1))

    boxed_match = list(_BOXED_RE.finditer(text))
    if boxed_match:
        return normalize_number(boxed_match[-1].group(1))

    marker_patterns = [
        r"(?:final\s+answer|answer)\s*(?:is|=|:)\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"所以答案是\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"答案[:：]?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
    ]
    for pattern in marker_patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches:
            return normalize_number(matches[-1].group(1))

    nums = _NUMBER_RE.findall(text)
    if not nums:
        return None
    return normalize_number(nums[-1])


def is_correct(response: str, gold: str | int | float) -> bool:
    return extract_answer(response) == normalize_number(gold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="JSONL with fields response and gold_answer")
    parser.add_argument("--output", type=Path, help="Output JSONL with extracted_answer and is_correct")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        cases = [
            ("We compute 12+7=19. #### 19", "19"),
            ("Final answer is $1,200.00", "1200"),
            ("Therefore \\boxed{42}", "42"),
            ("答案：-3.50", "-3.5"),
        ]
        for text, gold in cases:
            got = extract_answer(text)
            assert got == gold, (text, got, gold)
        print("answer_extractor self-test passed")
        return

    if not args.input or not args.output:
        raise SystemExit("Provide --input and --output, or use --self_test")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            response = row.get("response") or row.get("model_response") or ""
            gold = row.get("gold_answer") or row.get("answer")
            row["extracted_answer"] = extract_answer(response)
            row["gold_normalized"] = normalize_number(gold)
            row["is_correct"] = row["extracted_answer"] == row["gold_normalized"]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
