#!/usr/bin/env python3
"""
Select one evaluated sample per prompt from Geneval JSONL results.

Modes:
- sample0: keep sample index 0 (filename .../samples/00000.png) per prompt.
- bestofn: keep the best sample per prompt by Geneval outcome:
    1) prefer correct=True
    2) tie-break by fewer failed clauses (derived from reason lines)
    3) final tie-break by smaller sample index
"""

import argparse
import os
import re

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input Geneval JSONL results file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL with one sample per prompt")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sample0", "bestofn"],
        required=True,
        help="Selection mode",
    )
    return parser.parse_args()


def sample_index_from_filename(path: str) -> int:
    base = os.path.basename(path)
    m = re.match(r"(\d+)\.png$", base)
    if m is None:
        return 10**9
    return int(m.group(1))


def failure_count(reason: str) -> int:
    if not isinstance(reason, str):
        return 10**6
    stripped = reason.strip()
    if not stripped:
        return 0
    return len([line for line in stripped.splitlines() if line.strip()])


def select_sample0(group: pd.DataFrame) -> pd.Series:
    g = group.copy()
    g["sample_idx"] = g["filename"].astype(str).map(sample_index_from_filename)
    chosen = g[g["sample_idx"] == 0]
    if not chosen.empty:
        return chosen.iloc[0]
    # Fallback if sample 0 is absent: smallest available index.
    g = g.sort_values("sample_idx", ascending=True)
    return g.iloc[0]


def select_bestofn(group: pd.DataFrame) -> pd.Series:
    g = group.copy()
    g["sample_idx"] = g["filename"].astype(str).map(sample_index_from_filename)
    g["correct_int"] = g["correct"].astype(bool).astype(int)
    g["fail_count"] = g["reason"].map(failure_count)
    g = g.sort_values(
        by=["correct_int", "fail_count", "sample_idx"],
        ascending=[False, True, True],
    )
    return g.iloc[0]


def main():
    args = parse_args()
    df = pd.read_json(args.input, orient="records", lines=True)
    if len(df) == 0:
        raise ValueError(f"Input file has no rows: {args.input}")

    selector = select_sample0 if args.mode == "sample0" else select_bestofn
    selected_rows = [selector(group) for _, group in df.groupby("metadata", sort=False)]
    out_df = pd.DataFrame(selected_rows)

    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_json(args.output, orient="records", lines=True)
    print(f"Wrote {len(out_df)} prompts to {args.output} (mode={args.mode})")


if __name__ == "__main__":
    main()
