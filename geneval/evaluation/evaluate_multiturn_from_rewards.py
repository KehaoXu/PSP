#!/usr/bin/env python3
"""
Evaluate fixed multi-turn search policies on Geneval using collected reward trajectories.

This script:
1) Simulates scheduled or dynamic search from metrics.csv shards.
2) Recovers the selected final image per prompt from shard sample folders.
3) Builds a Geneval-compatible input folder.
4) Runs evaluate_images.py and summary_scores.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_int_list(value: str) -> List[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item) for item in value.split(",") if item.strip()]


@dataclass
class PromptTable:
    seeds: np.ndarray
    rewards: np.ndarray  # shape: [total_steps, num_seeds]
    shard_dir: Path


@dataclass
class ShardInfo:
    shard_dir: Path
    metrics_csv: Path
    prompt_start_id: int
    prompt_end_id: int
    time_steps: int


def read_geneval_metadata(path: Path) -> List[dict]:
    data: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    if not data:
        raise ValueError(f"No metadata entries found in {path}")
    return data


def discover_shards(rewards_root: Path) -> List[ShardInfo]:
    shards: List[ShardInfo] = []
    for shard_dir in sorted(rewards_root.glob("sd35_reward_signal_geneval_*")):
        if not shard_dir.is_dir():
            continue
        args_path = shard_dir / "args.json"
        metrics_csv = shard_dir / "metrics.csv"
        if not args_path.is_file() or not metrics_csv.is_file():
            continue
        with args_path.open("r", encoding="utf-8") as handle:
            args = json.load(handle)
        prompt_start = int(args["prompt_start_id"])
        prompt_end = int(args["prompt_end_id"])
        time_steps = int(args["time_steps"])
        shards.append(
            ShardInfo(
                shard_dir=shard_dir,
                metrics_csv=metrics_csv,
                prompt_start_id=prompt_start,
                prompt_end_id=prompt_end,
                time_steps=time_steps,
            )
        )
    if not shards:
        raise ValueError(f"No valid geneval shards found in {rewards_root}")
    return shards


def build_prompt_tables(shards: List[ShardInfo], total_steps: int) -> Dict[int, PromptTable]:
    prompt_tables: Dict[int, PromptTable] = {}

    for shard in shards:
        if shard.time_steps < total_steps:
            raise ValueError(
                f"Shard {shard.shard_dir} has time_steps={shard.time_steps} but total_steps={total_steps}"
            )

        df = pd.read_csv(
            shard.metrics_csv,
            usecols=["prompt_id", "step", "seed", "image_reward"],
        )
        df = df[df["step"] <= total_steps].copy()

        dup_count = int(df.duplicated(subset=["prompt_id", "step", "seed"]).sum())
        if dup_count > 0:
            raise ValueError(
                f"{shard.metrics_csv} contains {dup_count} duplicate (prompt_id, step, seed) rows."
            )

        for prompt_id, prompt_group in df.groupby("prompt_id", sort=False):
            pid = int(prompt_id)
            if pid in prompt_tables:
                raise ValueError(f"Prompt {pid} appears in multiple shards.")

            seeds = np.sort(prompt_group["seed"].unique().astype(int))
            pivot = prompt_group.pivot(index="step", columns="seed", values="image_reward")
            pivot = pivot.reindex(index=np.arange(1, total_steps + 1), columns=seeds)
            rewards = pivot.to_numpy(dtype=np.float64)

            if np.isnan(rewards).any():
                raise ValueError(
                    f"Prompt {pid} in {shard.shard_dir} has NaNs in reward table; "
                    "all seeds at all timesteps must be non-NaN."
                )

            prompt_tables[pid] = PromptTable(seeds=seeds, rewards=rewards, shard_dir=shard.shard_dir)

    return prompt_tables


def validate_strategy_args(args: argparse.Namespace) -> None:
    if args.k_init < 1:
        raise ValueError("--k-init must be >= 1")
    if args.total_steps < 2:
        raise ValueError("--total-steps must be >= 2")

    args.cutoff_times_list = parse_int_list(args.cutoff_times)
    if not args.cutoff_times_list:
        raise ValueError("--cutoff-times cannot be empty")
    if args.cutoff_times_list != sorted(args.cutoff_times_list):
        raise ValueError("--cutoff-times must be strictly increasing")
    if any(t <= 0 or t >= args.total_steps for t in args.cutoff_times_list):
        raise ValueError("Each cutoff must satisfy 0 < t < total_steps")

    if args.search_mode == "scheduled":
        args.remaining_particles_list = parse_int_list(args.remaining_particles)
        if len(args.remaining_particles_list) != len(args.cutoff_times_list):
            raise ValueError("--remaining-particles length must match --cutoff-times")
        prev = args.k_init
        for keep in args.remaining_particles_list:
            if keep < 1 or keep >= prev:
                raise ValueError(
                    "Scheduled mode requires strictly decreasing positive remaining particles."
                )
            prev = keep
    else:
        args.eps_list_values = parse_float_list(args.eps_list)
        if len(args.eps_list_values) != len(args.cutoff_times_list):
            raise ValueError("--eps-list length must match --cutoff-times")
        if args.budget < 1:
            raise ValueError("--budget must be >= 1")

        first_cutoff = args.cutoff_times_list[0]
        min_required = args.k_init * first_cutoff + (args.total_steps - first_cutoff)
        if min_required > args.budget:
            raise ValueError(
                "Invalid dynamic setup: cannot carry K to first cutoff and at least one to end "
                f"under budget={args.budget} (required={min_required})."
            )


def initial_survivors_from_seed(seeds: np.ndarray, k_init: int, seed_arg: int) -> List[int]:
    start = seed_arg * k_init
    end = start + k_init
    if end > len(seeds):
        raise ValueError(
            f"Seed index window out of bounds: requested indices [{start}, {end}) "
            f"for K={k_init}, but prompt has only {len(seeds)} available particles."
        )
    return [int(s) for s in seeds[start:end]]


def simulate_scheduled(
    *,
    table: PromptTable,
    k_init: int,
    cutoff_times: List[int],
    remaining_particles: List[int],
    total_steps: int,
    seed_arg: int,
) -> int:
    seed_to_col = {int(seed): idx for idx, seed in enumerate(table.seeds.tolist())}
    current = initial_survivors_from_seed(table.seeds, k_init, seed_arg)

    for step, keep_k in zip(cutoff_times, remaining_particles):
        row = table.rewards[step - 1]
        current_cols = [seed_to_col[s] for s in current]
        step_vals = row[current_cols]
        order = np.argsort(step_vals)[::-1]
        keep = max(1, min(int(keep_k), len(current)))
        current = [current[int(i)] for i in order[:keep]]

    final_row = table.rewards[total_steps - 1]
    final_vals = final_row[[seed_to_col[s] for s in current]]
    best_idx = int(np.argmax(final_vals))
    return int(current[best_idx])


def simulate_dynamic(
    *,
    table: PromptTable,
    k_init: int,
    cutoff_times: List[int],
    eps_list: List[float],
    budget: int,
    total_steps: int,
    seed_arg: int,
) -> int:
    seed_to_col = {int(seed): idx for idx, seed in enumerate(table.seeds.tolist())}
    current = initial_survivors_from_seed(table.seeds, k_init, seed_arg)

    cutoff_to_eps = {int(t): float(e) for t, e in zip(cutoff_times, eps_list)}
    next_cutoff_by_step: Dict[int, int] = {}
    for i, t in enumerate(cutoff_times):
        next_t = cutoff_times[i + 1] if i + 1 < len(cutoff_times) else total_steps
        next_cutoff_by_step[int(t)] = int(next_t)

    consumed = 0

    for step in range(1, total_steps + 1):
        survivors_now = len(current)
        consumed += survivors_now
        remaining_budget = budget - consumed

        row = table.rewards[step - 1]
        current_cols = [seed_to_col[s] for s in current]
        step_vals = row[current_cols]
        order = np.argsort(step_vals)[::-1]
        ordered_seeds = [current[int(i)] for i in order]
        ordered_vals = step_vals[order]

        if step == total_steps:
            break

        if step not in cutoff_to_eps:
            continue

        if len(ordered_seeds) > 1:
            eps = cutoff_to_eps[step]
            top_val = float(ordered_vals[0])
            keep_mask = ordered_vals >= (top_val - eps)
            kept = [seed for seed, keep in zip(ordered_seeds, keep_mask.tolist()) if keep]
            if len(kept) == 0:
                kept = [ordered_seeds[0]]
            current = kept

        next_cutoff = next_cutoff_by_step[int(step)]
        seg_len = next_cutoff - step
        tail_len = total_steps - next_cutoff
        required_if_keep_all = len(current) * seg_len + tail_len

        if required_if_keep_all > remaining_budget:
            remain_to_end = total_steps - step
            keep_n = 1 if remain_to_end <= 0 else int(remaining_budget // remain_to_end)
            if keep_n < 1:
                raise ValueError(
                    "Dynamic run became infeasible under provided budget at "
                    f"step={step}, prompt window seed={seed_arg}."
                )
            keep_n = min(keep_n, len(current))
            current_cols2 = [seed_to_col[s] for s in current]
            step_vals2 = row[current_cols2]
            order2 = np.argsort(step_vals2)[::-1]
            current = [current[int(i)] for i in order2[:keep_n]]

    final_row = table.rewards[total_steps - 1]
    final_vals = final_row[[seed_to_col[s] for s in current]]
    best_idx = int(np.argmax(final_vals))
    return int(current[best_idx])


def resolve_run_name(args: argparse.Namespace) -> str:
    if args.output_name:
        return args.output_name
    if args.search_mode == "scheduled":
        ts = "-".join(str(x) for x in args.cutoff_times_list)
        ks = "-".join(str(x) for x in args.remaining_particles_list)
        return f"scheduled_k{args.k_init}_ts{ts}_ks{ks}_t{args.total_steps}"
    ts = "-".join(str(x) for x in args.cutoff_times_list)
    eps = "-".join(f"{x:g}" for x in args.eps_list_values)
    return f"dynamic_k{args.k_init}_ts{ts}_eps{eps}_b{args.budget}_t{args.total_steps}"


def build_geneval_input(
    *,
    metadata: List[dict],
    selected_seed_by_prompt: Dict[int, int],
    prompt_tables: Dict[int, PromptTable],
    temp_input_dir: Path,
) -> None:
    temp_input_dir.mkdir(parents=True, exist_ok=False)

    for prompt_id, meta in enumerate(metadata):
        if prompt_id not in selected_seed_by_prompt:
            raise ValueError(f"No selected seed for prompt_id={prompt_id}")
        if prompt_id not in prompt_tables:
            raise ValueError(f"No prompt table found for prompt_id={prompt_id}")

        chosen_seed = selected_seed_by_prompt[prompt_id]
        shard_dir = prompt_tables[prompt_id].shard_dir
        src_img = shard_dir / "samples" / f"{prompt_id:05d}" / f"{chosen_seed:05d}.png"
        if not src_img.is_file():
            raise FileNotFoundError(f"Selected image not found: {src_img}")

        prompt_out_dir = temp_input_dir / f"{prompt_id:05d}"
        samples_out_dir = prompt_out_dir / "samples"
        samples_out_dir.mkdir(parents=True, exist_ok=True)
        with (prompt_out_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle)
        shutil.copy2(src_img, samples_out_dir / "00000.png")


def run_geneval_eval(
    *,
    repo_root: Path,
    eval_input_dir: Path,
    result_jsonl: Path,
    model_path: str,
) -> None:
    eval_script = repo_root / "geneval" / "evaluation" / "evaluate_images.py"
    summary_script = repo_root / "geneval" / "evaluation" / "summary_scores.py"

    cmd_eval = [
        sys.executable,
        str(eval_script),
        str(eval_input_dir),
        "--outfile",
        str(result_jsonl),
        "--samples-dir",
        "samples",
        "--model-path",
        model_path,
    ]
    subprocess.run(cmd_eval, check=True, cwd=repo_root)

    cmd_summary = [sys.executable, str(summary_script), str(result_jsonl)]
    subprocess.run(cmd_summary, check=True, cwd=repo_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate scheduled/dynamic multi-turn search outputs on Geneval from collected rewards."
    )
    parser.add_argument(
        "--search-mode",
        type=str,
        choices=["scheduled", "dynamic"],
        required=True,
    )
    parser.add_argument("--k-init", type=int, required=True)
    parser.add_argument("--cutoff-times", type=str, required=True)
    parser.add_argument("--remaining-particles", type=str, default=None)
    parser.add_argument("--eps-list", type=str, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument(
        "--rewards-root",
        type=str,
        default="output/sd3.5_collect_rewards",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="geneval/results/sdv3.5_hyper_search",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional run folder name under results root. If omitted, derived from hyperparameters.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="geneval/objdet",
    )
    parser.add_argument(
        "--keep-eval-input",
        action="store_true",
        help="Keep temporary reconstructed evaluation input directory.",
    )

    args = parser.parse_args()
    if args.search_mode == "scheduled":
        if not args.remaining_particles:
            raise ValueError("--remaining-particles is required for scheduled mode")
    else:
        if not args.eps_list:
            raise ValueError("--eps-list is required for dynamic mode")
        if args.budget is None:
            raise ValueError("--budget is required for dynamic mode")
    return args


def main() -> None:
    args = parse_args()
    validate_strategy_args(args)

    repo_root = Path(__file__).resolve().parents[2]
    rewards_root = (repo_root / args.rewards_root).resolve()
    metadata_path = (repo_root / args.metadata_path).resolve()
    results_root = (repo_root / args.results_root).resolve()

    metadata = read_geneval_metadata(metadata_path)
    shards = discover_shards(rewards_root)
    prompt_tables = build_prompt_tables(shards, args.total_steps)

    # Ensure every metadata prompt exists in loaded reward data.
    for prompt_id in range(len(metadata)):
        if prompt_id not in prompt_tables:
            raise ValueError(f"Missing reward trajectory for prompt_id={prompt_id}")

    selected_seed_by_prompt: Dict[int, int] = {}
    for prompt_id in range(len(metadata)):
        table = prompt_tables[prompt_id]
        if args.search_mode == "scheduled":
            selected_seed = simulate_scheduled(
                table=table,
                k_init=args.k_init,
                cutoff_times=args.cutoff_times_list,
                remaining_particles=args.remaining_particles_list,
                total_steps=args.total_steps,
                seed_arg=args.seed,
            )
        else:
            selected_seed = simulate_dynamic(
                table=table,
                k_init=args.k_init,
                cutoff_times=args.cutoff_times_list,
                eps_list=args.eps_list_values,
                budget=args.budget,
                total_steps=args.total_steps,
                seed_arg=args.seed,
            )
        selected_seed_by_prompt[prompt_id] = selected_seed

    run_name = resolve_run_name(args)
    seed_dir = results_root / run_name / f"seed={args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=False)
    result_jsonl = seed_dir / "results_samples.jsonl"
    args_out = seed_dir / "args.json"

    args_to_save = vars(args).copy()
    args_to_save["resolved_run_name"] = run_name
    args_to_save["num_prompts"] = len(metadata)
    with args_out.open("w", encoding="utf-8") as handle:
        json.dump(args_to_save, handle, indent=2)

    eval_input_dir = seed_dir / "_eval_input"
    build_geneval_input(
        metadata=metadata,
        selected_seed_by_prompt=selected_seed_by_prompt,
        prompt_tables=prompt_tables,
        temp_input_dir=eval_input_dir,
    )

    run_geneval_eval(
        repo_root=repo_root,
        eval_input_dir=eval_input_dir,
        result_jsonl=result_jsonl,
        model_path=args.model_path,
    )

    if not args.keep_eval_input:
        shutil.rmtree(eval_input_dir, ignore_errors=True)

    print(f"Saved evaluation results: {result_jsonl}")


if __name__ == "__main__":
    main()
