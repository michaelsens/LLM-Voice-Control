#!/usr/bin/env python3

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List


INPUT_PATH: Path = Path("expanded-2.jsonl")
OUTPUT_PATH: Path = INPUT_PATH.with_name("adjusted_data.jsonl")

NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
    21: "twenty one",
    22: "twenty two",
    23: "twenty three",
    24: "twenty four",
    25: "twenty five",
}
WORD_TO_NUM = {v: k for k, v in NUMBER_WORDS.items()}

# Numeric tokens (e.g. "25")
RE_NUMERIC = re.compile(r"\b([1-9][0-9]*)\b")
# Word tokens (e.g. "twenty five") – longest‑first to avoid partial matches
RE_NUMBER_WORDS = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(WORD_TO_NUM, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

# Remove *surrounding* single quotes but keep apostrophes in contractions
RE_SURROUNDING_QUOTES = re.compile(r"(?<!\w)'([^']+?)'(?!\w)")

# For duplicating static methods
STATIC_METHOD_COUNTS = {"openTab": 10, "closeTab": 10, "reload": 7}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_records(path: Path, records: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")


def replace_number_in_utterance(utterance: str, new_number: int) -> str:
    """Replace the *first* detected number (numeric or word) with *new_number*."""
    # Numeric first
    m = RE_NUMERIC.search(utterance)
    if m:
        return utterance[: m.start()] + str(new_number) + utterance[m.end() :]

    # Word variant
    m = RE_NUMBER_WORDS.search(utterance)
    if m:
        word = m.group(1)
        replacement = NUMBER_WORDS[new_number]
        # Preserve initial uppercase if it existed.
        if word[0].isupper():
            replacement = replacement.title()
        return utterance[: m.start()] + replacement + utterance[m.end() :]

    return utterance  # fallback (should not occur)


def remove_surrounding_quotes(text: str) -> str:
    """Delete single quotes used as quotation marks; keep apostrophes within words."""

    def _sub(match: re.Match):
        return match.group(1)  # inner content sans quotes

    return RE_SURROUNDING_QUOTES.sub(_sub, text)


def lowercase_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def deep_copy(obj):
    """Cheap deep‑copy via JSON encode/decode."""
    return json.loads(json.dumps(obj))

# ---------------------------------------------------------------------------
# Transformation steps
# ---------------------------------------------------------------------------

def expand_numeric_methods(records: List[Dict]) -> List[Dict]:
    expanded: List[Dict] = []

    scroll_extra = [5, 10, 15, 20, 25, 30, 50, 75, 150, 250, 750]
    scroll_range = list(range(100, 2001, 100)) + scroll_extra

    for rec in records:
        method = rec["rpc"]["method"]
        utterance = rec["utterance"]
        has_number = bool(RE_NUMERIC.search(utterance) or RE_NUMBER_WORDS.search(utterance))

        if method == "switchTab" and has_number:
            for i in range(1, 26):
                new = deep_copy(rec)
                new["utterance"] = replace_number_in_utterance(utterance, i)
                new["rpc"]["params"]["index"] = i - 1  # zero‑based
                expanded.append(new)
            continue  # skip original after duplication

        if method in {"goBack", "goForward"} and has_number:
            for i in range(1, 16):
                new = deep_copy(rec)
                new["utterance"] = replace_number_in_utterance(utterance, i)
                new["rpc"]["params"]["steps"] = i
                expanded.append(new)
            continue

        if method == "scroll" and has_number:
            for amount in scroll_range:
                new = deep_copy(rec)
                new["utterance"] = replace_number_in_utterance(utterance, amount)
                new["rpc"]["params"]["amount"] = amount
                expanded.append(new)
            continue

        expanded.append(rec)  # methods without special handling

    return expanded


def duplicate_static_methods(records: List[Dict]) -> List[Dict]:
    duped: List[Dict] = []
    for rec in records:
        times = STATIC_METHOD_COUNTS.get(rec["rpc"]["method"], 1)
        for _ in range(times):
            duped.append(deep_copy(rec))
    return duped


def clean_records(records: List[Dict]) -> List[Dict]:
    cleaned: List[Dict] = []
    for rec in records:
        rec = deep_copy(rec)
        utter = rec["utterance"]
        utter = remove_surrounding_quotes(utter)
        utter = lowercase_first(utter)
        rec["utterance"] = utter
        cleaned.append(rec)
    return cleaned


def deduplicate(records: List[Dict]) -> List[Dict]:
    seen: set[str] = set()
    unique: List[Dict] = []
    for rec in records:
        key = json.dumps(rec, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique


def transform(records: List[Dict]) -> List[Dict]:
    records = expand_numeric_methods(records)
    records = duplicate_static_methods(records)
    records = clean_records(records)
    records = deduplicate(records)
    random.shuffle(records)
    return records

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(f"Input file '{INPUT_PATH}' not found. Please update INPUT_PATH.")

    raw_records = load_records(INPUT_PATH)
    final_records = transform(raw_records)
    save_records(OUTPUT_PATH, final_records)
    print(f"Wrote {len(final_records)} records to '{OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()
