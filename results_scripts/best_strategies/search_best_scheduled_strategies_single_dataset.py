#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search scheduled strategies on a single dataset using explicit seed windows."
    )
    parser.add_argument("--model-label", type=str, default="sd35")
    parser.add_argument("--dataset-label", type=str, required=True)
    parser.add_argument("--metrics-csv", type=str, required=True)
    parser.add_argument("--geneval-csv", type=str, required=True)
    parser.add_argument(
        "--guidance-metric",
        type=str,
        choices=["image_reward", "human_preference"],
        default="image_reward",
        help="Metric used to rank strategies.",
    )
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--best-of-n", type=int, default=4)
    parser.add_argument("--max-cutoffs", type=int, default=4)
    parser.add_argument("--total-steps", type=int, default=32)
    parser.add_argument(
        "--possible-cutoff-times",
        type=str,
        default="2,4,6,8,10,12,14,16,18,20,22,24,26,28,30",
    )
    parser.add_argument(
        "--possible-remaining-seeds",
        type=str,
        default="32,28,24,20,16,12,8,4,3,2,1",
    )
    parser.add_argument("--output-json", type=str, required=True)
    return parser.parse_args()


@dataclass(frozen=True)
class ScheduledStrategy:
    k_init: int
    cutoff_times: tuple[int, ...]
    remaining_particles: tuple[int, ...]
    budget_steps: int
    total_particle_steps: int


def compute_total_particle_steps(
    k_init: int, cutoff_times: tuple[int, ...], remaining_particles: tuple[int, ...], total_steps: int
) -> int:
    steps = k_init * cutoff_times[0]
    for i in range(len(cutoff_times) - 1):
        steps += remaining_particles[i] * (cutoff_times[i + 1] - cutoff_times[i])
    steps += remaining_particles[-1] * (total_steps - cutoff_times[-1])
    return int(steps)


def build_remaining_sequences(
    possible_remaining: list[int], k_init: int, num_cutoffs: int
) -> list[tuple[int, ...]]:
    candidates = [k for k in possible_remaining if k < k_init]
    out: list[tuple[int, ...]] = []

    def dfs(prefix: list[int], last: int) -> None:
        if len(prefix) == num_cutoffs:
            out.append(tuple(prefix))
            return
        for k in candidates:
            if k < last:
                prefix.append(k)
                dfs(prefix, k)
                prefix.pop()

    dfs([], k_init)
    return out


def enumerate_strategies(
    *,
    total_steps: int,
    max_cutoffs: int,
    cutoff_candidates: list[int],
    possible_remaining: list[int],
    best_of_n: int,
) -> list[ScheduledStrategy]:
    budget = best_of_n * total_steps
    strategies: list[ScheduledStrategy] = []

    valid_cutoffs = sorted([t for t in cutoff_candidates if 0 < t < total_steps])
    for k_init in sorted(set(possible_remaining), reverse=True):
        for n_cut in range(1, max_cutoffs + 1):
            rem_sequences = build_remaining_sequences(possible_remaining, k_init, n_cut)
            if not rem_sequences:
                continue
            for ts in combinations(valid_cutoffs, n_cut):
                ts_t = tuple(int(x) for x in ts)
                for rs in rem_sequences:
                    total_ps = compute_total_particle_steps(k_init, ts_t, rs, total_steps)
                    if total_ps <= budget:
                        strategies.append(
                            ScheduledStrategy(
                                k_init=int(k_init),
                                cutoff_times=ts_t,
                                remaining_particles=tuple(int(x) for x in rs),
                                budget_steps=int(budget),
                                total_particle_steps=int(total_ps),
                            )
                        )
    return strategies


def load_data(metrics_csv: Path, geneval_csv: Path, total_steps: int, dataset_label: str) -> pd.DataFrame:
    if metrics_csv.is_dir():
        all_metric_files = sorted(metrics_csv.rglob("metrics.csv"))
        if dataset_label == "benchmark_ir":
            metric_files = [p for p in all_metric_files if "benchmark" in str(p).lower()]
        elif dataset_label == "geneval":
            metric_files = [p for p in all_metric_files if "geneval" in str(p).lower()]
        else:
            metric_files = all_metric_files
        if not metric_files:
            raise ValueError(
                f"No metrics.csv files found under {metrics_csv} for dataset_label={dataset_label}."
            )
        metrics = pd.concat([pd.read_csv(p) for p in metric_files], ignore_index=True)
    else:
        metrics = pd.read_csv(metrics_csv)
    req_metrics = {"prompt_id", "seed", "step", "image_reward", "human_preference"}
    miss_metrics = req_metrics - set(metrics.columns)
    if miss_metrics:
        raise ValueError(f"Missing metrics columns: {sorted(miss_metrics)}")
    metrics = metrics[list(req_metrics)].copy()
    metrics = metrics[metrics["step"] <= total_steps].copy()
    metrics = metrics.drop_duplicates(subset=["prompt_id", "seed", "step"], keep="last")

    geneval = pd.read_csv(geneval_csv)
    req_geneval = {"prompt_id", "seed", "correct", "tag"}
    miss_geneval = req_geneval - set(geneval.columns)
    if miss_geneval:
        raise ValueError(f"Missing geneval columns: {sorted(miss_geneval)}")
    geneval = geneval[list(req_geneval)].copy()
    geneval["correct"] = geneval["correct"].astype(float)

    merged = metrics.merge(geneval, on=["prompt_id", "seed"], how="inner")
    if merged.empty:
        raise RuntimeError("No overlap between metrics CSV and geneval CSV by (prompt_id, seed).")
    return merged


def add_local_seed_idx(df: pd.DataFrame) -> pd.DataFrame:
    pair = df[["prompt_id", "seed"]].drop_duplicates().sort_values(["prompt_id", "seed"]).reset_index(drop=True)
    pair["local_seed_idx"] = pair.groupby("prompt_id").cumcount()
    return df.merge(pair, on=["prompt_id", "seed"], how="inner")


def overall_geneval_from_selected(selected: pd.DataFrame) -> float:
    if selected.empty:
        return float("nan")
    return float(selected.groupby("tag")["correct"].mean().mean())


def scheduled_select_prompt_seed(
    grp: pd.DataFrame,
    *,
    k_init: int,
    logical_seed: int,
    cutoff_times: tuple[int, ...],
    remaining_particles: tuple[int, ...],
    guidance_col: str,
    total_steps: int,
) -> int:
    start = logical_seed * k_init
    end = (logical_seed + 1) * k_init - 1
    pool = grp[(grp["local_seed_idx"] >= start) & (grp["local_seed_idx"] <= end)]
    if pool.empty:
        raise ValueError(
            f"Empty seed window {start}..{end} for prompt_id={int(grp['prompt_id'].iloc[0])}. "
            "Search space K is incompatible with available particles."
        )
    survivors = sorted(pool["seed"].unique().astype(int).tolist())
    for cutoff, keep in zip(cutoff_times, remaining_particles):
        at_cutoff = grp[(grp["step"] == cutoff) & (grp["seed"].isin(survivors))]
        at_cutoff = at_cutoff.sort_values([guidance_col, "seed"], ascending=[False, True])
        survivors = at_cutoff["seed"].head(int(keep)).astype(int).tolist()

    at_final = grp[(grp["step"] == total_steps) & (grp["seed"].isin(survivors))]
    at_final = at_final.sort_values([guidance_col, "seed"], ascending=[False, True])
    if at_final.empty:
        raise ValueError(
            f"Missing final-step survivors for prompt_id={int(grp['prompt_id'].iloc[0])}."
        )
    return int(at_final.iloc[0]["seed"])


def evaluate_strategy(
    df: pd.DataFrame,
    strategy: ScheduledStrategy,
    seeds: list[int],
    total_steps: int,
    guidance_metric: str,
) -> dict:
    prompts = sorted(df["prompt_id"].unique().tolist())
    per_seed_overall: dict[str, dict] = {}
    per_prompt: dict[str, dict] = {str(int(pid)): {"seed_results": {}} for pid in prompts}
    guidance_col = "image_reward" if guidance_metric == "image_reward" else "human_preference"

    for logical_seed in seeds:
        selected_rows = []
        for prompt_id, grp in df.groupby("prompt_id", sort=False):
            chosen_seed = scheduled_select_prompt_seed(
                grp,
                k_init=strategy.k_init,
                logical_seed=logical_seed,
                cutoff_times=strategy.cutoff_times,
                remaining_particles=strategy.remaining_particles,
                guidance_col=guidance_col,
                total_steps=total_steps,
            )
            row = grp[(grp["step"] == total_steps) & (grp["seed"] == chosen_seed)].iloc[0]
            selected_rows.append(row)

            per_prompt[str(int(prompt_id))]["seed_results"][str(logical_seed)] = {
                "IR": float(row["image_reward"]),
                "HPS": float(row["human_preference"]),
                "GenEvalOverall": float(row["correct"]),
            }

        sel = pd.DataFrame(selected_rows)
        per_seed_overall[str(logical_seed)] = {
            "IR": float(sel["image_reward"].mean()),
            "HPS": float(sel["human_preference"].mean()),
            "GenEvalOverall": overall_geneval_from_selected(sel),
        }

    seed_df = pd.DataFrame(per_seed_overall).T
    seed_mean = {k: float(seed_df[k].mean()) for k in ["IR", "HPS", "GenEvalOverall"]}

    for prompt_id in per_prompt.keys():
        prompt_df = pd.DataFrame(per_prompt[prompt_id]["seed_results"]).T
        per_prompt[prompt_id]["seed_mean"] = {
            k: float(prompt_df[k].mean()) for k in ["IR", "HPS", "GenEvalOverall"]
        }

    return {
        "seed_results": per_seed_overall,
        "seed_mean": seed_mean,
        "per_prompt": per_prompt,
    }


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    cutoffs = parse_int_list(args.possible_cutoff_times)
    possible_remaining = parse_int_list(args.possible_remaining_seeds)
    if len(seeds) < 1:
        raise ValueError("At least one seed must be provided.")

    merged = load_data(
        Path(args.metrics_csv),
        Path(args.geneval_csv),
        args.total_steps,
        args.dataset_label,
    )
    traj = merged[["prompt_id", "seed", "step"]].drop_duplicates()
    seeds_per_prompt = traj.groupby("prompt_id")["seed"].nunique()
    steps_per_prompt = traj.groupby("prompt_id")["step"].nunique()
    if seeds_per_prompt.nunique() != 1:
        raise ValueError(
            "Inconsistent seeds per prompt detected. "
            f"min={int(seeds_per_prompt.min())}, max={int(seeds_per_prompt.max())}"
        )
    if steps_per_prompt.nunique() != 1:
        raise ValueError(
            "Inconsistent total timesteps per prompt detected. "
            f"min={int(steps_per_prompt.min())}, max={int(steps_per_prompt.max())}"
        )
    print(
        f"Trajectory check: seeds_per_prompt={int(seeds_per_prompt.iloc[0])} "
        f"T={int(steps_per_prompt.iloc[0])}"
    )
    merged = add_local_seed_idx(merged)
    total_inference_paths = int(merged[["prompt_id", "seed"]].drop_duplicates().shape[0])
    total_distinct_prompts = int(merged["prompt_id"].nunique())
    print(
        f"Sanity check: inference_paths={total_inference_paths} "
        f"distinct_prompts={total_distinct_prompts}"
    )

    strategies = enumerate_strategies(
        total_steps=args.total_steps,
        max_cutoffs=args.max_cutoffs,
        cutoff_candidates=cutoffs,
        possible_remaining=possible_remaining,
        best_of_n=args.best_of_n,
    )
    if not strategies:
        raise ValueError("No valid scheduled strategies found with current constraints.")

    entries = []
    total_valid_strategies = len(strategies)
    print(f"Evaluating {total_valid_strategies} valid scheduled strategies...")
    for strategy in tqdm(
        strategies,
        total=total_valid_strategies,
        desc="Scheduled strategy search",
        unit="strategy",
    ):
        eval_out = evaluate_strategy(merged, strategy, seeds, args.total_steps, args.guidance_metric)
        entries.append(
            {
                "strategy": {
                    "type": "scheduled",
                    "k_init": strategy.k_init,
                    "cutoff_times": list(strategy.cutoff_times),
                    "remaining_particles": list(strategy.remaining_particles),
                    "budget_steps": strategy.budget_steps,
                    "total_particle_steps": strategy.total_particle_steps,
                },
                "seed_results": eval_out["seed_results"],
                "seed_mean": eval_out["seed_mean"],
                "per_prompt": eval_out["per_prompt"],
            }
        )

    rank_key = "IR" if args.guidance_metric == "image_reward" else "HPS"
    entries = sorted(entries, key=lambda x: x["seed_mean"][rank_key], reverse=True)
    for i, item in enumerate(entries, start=1):
        item["rank"] = i

    out = {
        "meta": {
            "model_label": args.model_label,
            "dataset_label": args.dataset_label,
            "guidance_metric": args.guidance_metric,
            "rank_metric": rank_key,
            "seeds": seeds,
            "best_of_n_budget": args.best_of_n,
            "max_cutoffs": args.max_cutoffs,
            "total_steps": args.total_steps,
            "possible_cutoff_times": cutoffs,
            "possible_remaining_seeds": possible_remaining,
            "num_prompts": int(merged["prompt_id"].nunique()),
            "num_strategies": len(entries),
            "metrics_csv": str(Path(args.metrics_csv).resolve()),
            "geneval_csv": str(Path(args.geneval_csv).resolve()),
        },
        "strategies": entries,
    }

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved JSON: {out_path}")


if __name__ == "__main__":
    main()
