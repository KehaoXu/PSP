#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute regular inference vs best-of-n from precomputed reward signal."
    )
    parser.add_argument(
        "--model-label",
        type=str,
        default="model",
        help="Label used in report titles (e.g., sd35, sdv15, sdxl).",
    )
    parser.add_argument(
        "--rewards-root",
        type=str,
        required=True,
        help="Root containing precomputed reward signal folders for the model.",
    )
    parser.add_argument(
        "--geneval-csv",
        type=str,
        required=True,
        help="Geneval per-sample cache CSV for the model.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=4,
        help="Best-of-n pool size.",
    )
    parser.add_argument(
        "--logical-seeds",
        type=str,
        default=None,
        help='Comma-separated logical seed indices (prompt-local), e.g. "0,1,2".',
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=3,
        help="Fallback when --logical-seeds is omitted: evaluates 0..num_seeds-1.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where summary txt files are written.",
    )
    parser.add_argument(
        "--guidance-metric",
        type=str,
        choices=["both", "ir", "hps"],
        default="both",
        help="Guidance metric(s) to evaluate for best-of-n selection.",
    )
    return parser.parse_args()


def parse_logical_seeds(args) -> list[int]:
    if args.logical_seeds is not None:
        vals = [s.strip() for s in args.logical_seeds.split(",") if s.strip()]
        if not vals:
            raise ValueError("--logical-seeds provided but empty.")
        return [int(v) for v in vals]
    return list(range(args.num_seeds))


def load_final_metrics(rewards_root: Path) -> pd.DataFrame:
    metric_paths = sorted(rewards_root.glob("*geneval*/metrics.csv"))
    if not metric_paths:
        raise FileNotFoundError(f"No geneval metrics.csv found under {rewards_root}")

    parts = [pd.read_csv(p) for p in metric_paths]
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["prompt_id", "seed", "step"]).drop_duplicates(
        subset=["prompt_id", "seed"], keep="last"
    )
    return df[["prompt_id", "seed", "image_reward", "human_preference"]]


def load_geneval(geneval_csv: Path) -> pd.DataFrame:
    if not geneval_csv.exists():
        raise FileNotFoundError(f"Geneval CSV not found: {geneval_csv}")
    df = pd.read_csv(geneval_csv)
    needed = {"prompt_id", "seed", "geneval_soft_score", "correct", "tag"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in geneval CSV: {sorted(missing)}")
    out = df[["prompt_id", "seed", "geneval_soft_score", "correct", "tag"]].copy()
    out["correct"] = out["correct"].astype(float)
    return out


def add_prompt_local_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["prompt_id", "seed"]).reset_index(drop=True).copy()
    out["local_seed_idx"] = out.groupby("prompt_id").cumcount()
    return out


def geneval_overall_score(selected: pd.DataFrame) -> float:
    if selected.empty:
        return float("nan")
    task_scores = selected.groupby("tag")["correct"].mean()
    return float(task_scores.mean()) if len(task_scores) else float("nan")


def regular_for_seed(merged: pd.DataFrame, logical_seed: int) -> dict:
    chosen = merged[merged["local_seed_idx"] == logical_seed].copy()
    overall = geneval_overall_score(chosen)
    return {
        "logical_seed": logical_seed,
        "particle_spec": f"{logical_seed}",
        "num_prompts": int(chosen["prompt_id"].nunique()),
        "mean_ir": float(chosen["image_reward"].mean()),
        "mean_hps": float(chosen["human_preference"].mean()),
        "overall_geneval": overall,
        "overall_geneval_from_ir": overall,
        "overall_geneval_from_hps": overall,
    }


def bestofn_for_seed(merged: pd.DataFrame, logical_seed: int, n: int, guidance_metric: str) -> dict:
    start = logical_seed * n
    end = (logical_seed + 1) * n - 1
    cur = merged[(merged["local_seed_idx"] >= start) & (merged["local_seed_idx"] <= end)].copy()
    if cur.empty:
        return {
            "logical_seed": logical_seed,
            "particle_spec": f"{start}..{end}",
            "num_prompts": 0,
            "mean_ir": float("nan"),
            "mean_hps": float("nan"),
            "overall_geneval": float("nan"),
            "overall_geneval_from_ir": float("nan"),
            "overall_geneval_from_hps": float("nan"),
        }

    idx_ir = cur.groupby("prompt_id")["image_reward"].idxmax()
    idx_geneval = cur.groupby("prompt_id")["geneval_soft_score"].idxmax()

    chosen_ir = cur.loc[idx_ir]
    chosen_hps = None
    if guidance_metric in {"both", "hps"}:
        idx_hps = cur.groupby("prompt_id")["human_preference"].idxmax()
        chosen_hps = cur.loc[idx_hps]
    chosen_geneval = cur.loc[idx_geneval]

    return {
        "logical_seed": logical_seed,
        "particle_spec": f"{start}..{end}",
        "num_prompts": int(chosen_ir["prompt_id"].nunique()),
        "mean_ir": float(chosen_ir["image_reward"].mean()),
        "mean_hps": (
            float(chosen_hps["human_preference"].mean()) if chosen_hps is not None else float("nan")
        ),
        "overall_geneval": geneval_overall_score(chosen_geneval),
        "overall_geneval_from_ir": geneval_overall_score(chosen_ir),
        "overall_geneval_from_hps": (
            geneval_overall_score(chosen_hps) if chosen_hps is not None else float("nan")
        ),
    }


def summarize(rows):
    df = pd.DataFrame(rows)
    return {
        "mean_ir": float(df["mean_ir"].mean()),
        "mean_hps": float(df["mean_hps"].mean()),
        "overall_geneval": float(df["overall_geneval"].mean()),
        "overall_geneval_from_ir": float(df["overall_geneval_from_ir"].mean()),
        "overall_geneval_from_hps": float(df["overall_geneval_from_hps"].mean()),
    }


def format_report(title: str, args_dict: dict, rows: list, overall: dict) -> str:
    lines = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("Args")
    lines.append("----")
    lines.append(json.dumps(args_dict, indent=2, sort_keys=True))
    lines.append("")
    lines.append("Per-seed results")
    lines.append("----------------")
    for r in rows:
        lines.append(
            f"seed={r['logical_seed']:<2d} particles={r['particle_spec']:<10s} "
            f"prompts={r['num_prompts']:<4d} "
            f"IR={r['mean_ir']:.6f} HPS={r['mean_hps']:.6f} "
            f"GenEvalOverall={r['overall_geneval']:.6f} "
            f"GenEvalOverall@IR={r['overall_geneval_from_ir']:.6f} "
            f"GenEvalOverall@HPS={r['overall_geneval_from_hps']:.6f}"
        )
    lines.append("")
    lines.append("Mean across seeds")
    lines.append("-----------------")
    lines.append(
        f"IR={overall['mean_ir']:.6f} HPS={overall['mean_hps']:.6f} "
        f"GenEvalOverall={overall['overall_geneval']:.6f} "
        f"GenEvalOverall@IR={overall['overall_geneval_from_ir']:.6f} "
        f"GenEvalOverall@HPS={overall['overall_geneval_from_hps']:.6f}"
    )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    logical_seeds = parse_logical_seeds(args)
    rewards_root = Path(args.rewards_root).resolve()
    geneval_csv = Path(args.geneval_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    final_metrics = load_final_metrics(rewards_root)
    geneval = load_geneval(geneval_csv)
    merged = final_metrics.merge(geneval, on=["prompt_id", "seed"], how="inner")
    merged = add_prompt_local_index(merged)
    if merged.empty:
        raise RuntimeError("No overlapping rows between reward metrics and Geneval scores.")

    regular_rows = []
    best_rows = []
    for logical_seed in logical_seeds:
        regular_rows.append(regular_for_seed(merged, logical_seed))
        best_rows.append(bestofn_for_seed(merged, logical_seed, args.n, args.guidance_metric))

    regular_overall = summarize(regular_rows)
    best_overall = summarize(best_rows)

    args_dict = {
        "model_label": args.model_label,
        "rewards_root": str(rewards_root),
        "geneval_csv": str(geneval_csv),
        "n": args.n,
        "guidance_metric": args.guidance_metric,
        "logical_seeds": logical_seeds,
        "num_seeds": args.num_seeds,
        "output_dir": str(output_dir),
    }

    regular_report = format_report(
        f"{args.model_label} Regular Inference Baseline",
        args_dict=args_dict,
        rows=regular_rows,
        overall=regular_overall,
    )
    best_report = format_report(
        f"{args.model_label} Best-of-{args.n} Baseline",
        args_dict=args_dict,
        rows=best_rows,
        overall=best_overall,
    )

    regular_path = output_dir / "regular_inference.txt"
    best_path = output_dir / f"best_of_n={args.n}.txt"
    regular_path.write_text(regular_report, encoding="utf-8")
    best_path.write_text(best_report, encoding="utf-8")

    print(regular_report)
    print(best_report)
    print(f"Saved: {regular_path}")
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()
