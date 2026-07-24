#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd


def parse_int_list(value: str) -> list[int]:
    vals = [v.strip() for v in value.split(",") if v.strip()]
    if not vals:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return [int(v) for v in vals]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate MT scheduled strategy from precomputed reward trajectories."
    )
    parser.add_argument("--model-label", type=str, default="model")
    parser.add_argument("--rewards-root", type=str, required=True)
    parser.add_argument("--geneval-csv", type=str, required=True)
    parser.add_argument("--k-init", type=int, default=8, help="Initial pool size K.")
    parser.add_argument(
        "--cutoff-times",
        type=str,
        default="8,16",
        help='Comma-separated cutoff steps, e.g. "8,16".',
    )
    parser.add_argument(
        "--remaining-particles",
        type=str,
        default="4,2",
        help='Comma-separated remaining particles at each cutoff, e.g. "4,2".',
    )
    parser.add_argument("--total-steps", type=int, default=32, help="Final step T.")
    parser.add_argument(
        "--logical-seeds",
        type=str,
        default="0,1,2",
        help='Comma-separated logical seed indices (prompt-local), e.g. "0,1,2".',
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=3,
        help="Fallback when --logical-seeds is omitted: evaluates 0..num_seeds-1.",
    )
    parser.add_argument(
        "--guidance-metric",
        type=str,
        choices=["both", "ir", "hps"],
        default="both",
        help="Guidance metric(s) to evaluate for MT scheduled selection.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def validate_args(args) -> tuple[list[int], list[int], list[int]]:
    if args.k_init < 1:
        raise ValueError("--k-init must be >= 1.")
    if args.total_steps < 2:
        raise ValueError("--total-steps must be >= 2.")

    logical_seeds = (
        parse_int_list(args.logical_seeds)
        if args.logical_seeds is not None
        else list(range(args.num_seeds))
    )
    cutoff_times = parse_int_list(args.cutoff_times)
    remaining_particles = parse_int_list(args.remaining_particles)

    if len(cutoff_times) != len(remaining_particles):
        raise ValueError("--cutoff-times and --remaining-particles must have equal length.")
    if cutoff_times != sorted(cutoff_times):
        raise ValueError("--cutoff-times must be strictly increasing.")
    if any(t <= 0 or t >= args.total_steps for t in cutoff_times):
        raise ValueError("Each cutoff must satisfy 0 < cutoff < total_steps.")

    prev = args.k_init
    for keep in remaining_particles:
        if keep < 1 or keep >= prev:
            raise ValueError(
                "Scheduled setup must be strictly decreasing positive survivors at each cutoff."
            )
        prev = keep

    return logical_seeds, cutoff_times, remaining_particles


def load_reward_trajectories(rewards_root: Path, total_steps: int) -> pd.DataFrame:
    metric_paths = sorted(rewards_root.glob("*geneval*/metrics.csv"))
    if not metric_paths:
        raise FileNotFoundError(f"No geneval metrics.csv found under {rewards_root}")

    parts = []
    cols = ["prompt_id", "seed", "step", "image_reward", "human_preference"]
    for p in metric_paths:
        df = pd.read_csv(p, usecols=cols)
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    out = out[out["step"] <= total_steps].copy()
    out = out.drop_duplicates(subset=["prompt_id", "seed", "step"], keep="last")
    return out


def load_geneval(geneval_csv: Path) -> pd.DataFrame:
    if not geneval_csv.exists():
        raise FileNotFoundError(f"Geneval CSV not found: {geneval_csv}")
    df = pd.read_csv(geneval_csv)
    needed = {"prompt_id", "seed", "correct", "tag"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in geneval CSV: {sorted(missing)}")
    out = df[["prompt_id", "seed", "correct", "tag"]].copy()
    out["correct"] = out["correct"].astype(float)
    return out


def add_prompt_local_index(traj: pd.DataFrame) -> pd.DataFrame:
    pair_idx = (
        traj[["prompt_id", "seed"]]
        .drop_duplicates()
        .sort_values(["prompt_id", "seed"])
        .reset_index(drop=True)
    )
    pair_idx["local_seed_idx"] = pair_idx.groupby("prompt_id").cumcount()
    return traj.merge(pair_idx, on=["prompt_id", "seed"], how="inner")


def geneval_overall_score(selected_pairs: pd.DataFrame) -> float:
    if selected_pairs.empty:
        return float("nan")
    task_scores = selected_pairs.groupby("tag")["correct"].mean()
    return float(task_scores.mean()) if len(task_scores) else float("nan")


def select_final_pairs_for_guidance(
    traj: pd.DataFrame,
    *,
    logical_seed: int,
    k_init: int,
    cutoff_times: list[int],
    remaining_particles: list[int],
    total_steps: int,
    guidance_col: str,
) -> pd.DataFrame:
    start = logical_seed * k_init
    end = (logical_seed + 1) * k_init - 1
    cur = traj[(traj["local_seed_idx"] >= start) & (traj["local_seed_idx"] <= end)].copy()
    if cur.empty:
        raise ValueError(
            f"No particles in window {start}..{end} for logical_seed={logical_seed}. "
            "Check k-init/logical-seeds against available particles."
        )

    chosen_rows = []
    for prompt_id, grp in cur.groupby("prompt_id", sort=False):
        survivors = sorted(grp["seed"].unique().tolist())
        for cutoff, keep in zip(cutoff_times, remaining_particles):
            at_cutoff = grp[(grp["step"] == cutoff) & (grp["seed"].isin(survivors))]
            if at_cutoff.empty:
                raise ValueError(
                    f"Missing cutoff step={cutoff} values for prompt_id={prompt_id}, "
                    f"logical_seed={logical_seed}, guidance={guidance_col}."
                )
            at_cutoff = at_cutoff.sort_values([guidance_col, "seed"], ascending=[False, True])
            survivors = at_cutoff["seed"].head(keep).astype(int).tolist()

        at_final = grp[(grp["step"] == total_steps) & (grp["seed"].isin(survivors))]
        if at_final.empty:
            raise ValueError(
                f"Missing final step={total_steps} values for prompt_id={prompt_id}, "
                f"logical_seed={logical_seed}, guidance={guidance_col}."
            )
        best_final = at_final.sort_values([guidance_col, "seed"], ascending=[False, True]).iloc[0]
        chosen_rows.append(best_final)

    return pd.DataFrame(chosen_rows)


def simulate_for_seed(
    traj: pd.DataFrame,
    geneval_pairs: pd.DataFrame,
    *,
    logical_seed: int,
    k_init: int,
    cutoff_times: list[int],
    remaining_particles: list[int],
    total_steps: int,
    guidance_metric: str,
) -> dict:
    selected_ir = None
    selected_hps = None
    ir_geneval = None
    hps_geneval = None
    if guidance_metric in {"both", "ir"}:
        selected_ir = select_final_pairs_for_guidance(
            traj,
            logical_seed=logical_seed,
            k_init=k_init,
            cutoff_times=cutoff_times,
            remaining_particles=remaining_particles,
            total_steps=total_steps,
            guidance_col="image_reward",
        )
        ir_geneval = selected_ir[["prompt_id", "seed"]].merge(
            geneval_pairs, on=["prompt_id", "seed"], how="inner"
        )
    if guidance_metric in {"both", "hps"}:
        selected_hps = select_final_pairs_for_guidance(
            traj,
            logical_seed=logical_seed,
            k_init=k_init,
            cutoff_times=cutoff_times,
            remaining_particles=remaining_particles,
            total_steps=total_steps,
            guidance_col="human_preference",
        )
        hps_geneval = selected_hps[["prompt_id", "seed"]].merge(
            geneval_pairs, on=["prompt_id", "seed"], how="inner"
        )

    start = logical_seed * k_init
    end = (logical_seed + 1) * k_init - 1
    return {
        "logical_seed": logical_seed,
        "particle_spec": f"{start}..{end}",
        "num_prompts": int(
            (selected_ir if selected_ir is not None else selected_hps)["prompt_id"].nunique()
        ),
        "mean_ir": (
            float(selected_ir["image_reward"].mean()) if selected_ir is not None else float("nan")
        ),
        "mean_hps": (
            float(selected_hps["human_preference"].mean()) if selected_hps is not None else float("nan")
        ),
        "overall_geneval_from_ir": (
            geneval_overall_score(ir_geneval) if ir_geneval is not None else float("nan")
        ),
        "overall_geneval_from_hps": (
            geneval_overall_score(hps_geneval) if hps_geneval is not None else float("nan")
        ),
    }


def summarize(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    return {
        "mean_ir": float(df["mean_ir"].mean()),
        "mean_hps": float(df["mean_hps"].mean()),
        "overall_geneval_from_ir": float(df["overall_geneval_from_ir"].mean()),
        "overall_geneval_from_hps": float(df["overall_geneval_from_hps"].mean()),
    }


def format_report(title: str, args_dict: dict, rows: list[dict], overall: dict) -> str:
    lines = [title, "=" * len(title), "", "Args", "----", json.dumps(args_dict, indent=2, sort_keys=True), ""]
    lines.extend(["Per-seed results", "----------------"])
    for r in rows:
        lines.append(
            f"seed={r['logical_seed']:<2d} particles={r['particle_spec']:<10s} "
            f"prompts={r['num_prompts']:<4d} "
            f"IR={r['mean_ir']:.6f} HPS={r['mean_hps']:.6f} "
            f"GenEvalOverall@IR={r['overall_geneval_from_ir']:.6f} "
            f"GenEvalOverall@HPS={r['overall_geneval_from_hps']:.6f}"
        )
    lines.extend(["", "Mean across seeds", "-----------------"])
    lines.append(
        f"IR={overall['mean_ir']:.6f} HPS={overall['mean_hps']:.6f} "
        f"GenEvalOverall@IR={overall['overall_geneval_from_ir']:.6f} "
        f"GenEvalOverall@HPS={overall['overall_geneval_from_hps']:.6f}"
    )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    logical_seeds, cutoff_times, remaining_particles = validate_args(args)

    rewards_root = Path(args.rewards_root).resolve()
    geneval_csv = Path(args.geneval_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    traj = load_reward_trajectories(rewards_root, args.total_steps)
    traj = add_prompt_local_index(traj)
    geneval_pairs = load_geneval(geneval_csv)

    rows = []
    for logical_seed in logical_seeds:
        rows.append(
            simulate_for_seed(
                traj,
                geneval_pairs,
                logical_seed=logical_seed,
                k_init=args.k_init,
                cutoff_times=cutoff_times,
                remaining_particles=remaining_particles,
                total_steps=args.total_steps,
                guidance_metric=args.guidance_metric,
            )
        )
    overall = summarize(rows)

    args_dict = {
        "model_label": args.model_label,
        "rewards_root": str(rewards_root),
        "geneval_csv": str(geneval_csv),
        "k_init": args.k_init,
        "cutoff_times": cutoff_times,
        "remaining_particles": remaining_particles,
        "total_steps": args.total_steps,
        "guidance_metric": args.guidance_metric,
        "logical_seeds": logical_seeds,
        "num_seeds": args.num_seeds,
        "output_dir": str(output_dir),
    }
    title = f"{args.model_label} MT Scheduled Baseline"
    report = format_report(title, args_dict, rows, overall)

    ts_str = "-".join(str(x) for x in cutoff_times)
    rs_str = "-".join(str(x) for x in remaining_particles)
    out_path = output_dir / f"mt_scheduled_k={args.k_init}_ts={ts_str}_r={rs_str}_t={args.total_steps}.txt"
    out_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
