#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from tqdm import tqdm


MODEL_CONFIG: dict[str, dict[str, Any]] = {
    "sd15": {
        "name": "Stable Diffusion v1.5",
        "rewards_root": "results/reward_signal/sd15_reward_signal",
        "geneval_csv": "results/reward_signal/sd15_reward_signal/sd15_geneval_sample_scores.csv",
        "total_steps": 64,
    },
    "sdxl": {
        "name": "Stable Diffusion XL",
        "rewards_root": "results/reward_signal/sdxl_reward_signal",
        "geneval_csv": "results/reward_signal/sdxl_reward_signal/sdxl_geneval_sample_scores.csv",
        "total_steps": 64,
    },
    "sd35": {
        "name": "Stable Diffusion 3.5",
        "rewards_root": "results/reward_signal/sd35_reward_signal",
        "geneval_csv": "results/reward_signal/sd35_reward_signal/sd35_geneval_sample_scores.csv",
        "total_steps": 32,
    },
}

MODEL_COLORS: dict[str, str] = {
    "sd15": "blue",
    "sdxl": "gold",
    "sd35": "green",
}
PLOT_MODELS: list[str] = ["sd15", "sdxl", "sd35"]


@dataclass
class Strategy:
    effective_n: int
    k_init: int
    cutoff_times: list[int]
    remaining_particles: list[int]


def parse_int_list(value: str) -> list[int]:
    vals = [x.strip() for x in value.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return [int(v) for v in vals]


def parse_str_list(value: str) -> list[str]:
    vals = [x.strip() for x in value.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected a non-empty comma-separated string list.")
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scale a base PSP strategy across effective N values, cache strategy "
            "scores, and plot IR/GenEval scaling curves."
        )
    )
    parser.add_argument(
        "--models",
        type=str,
        default="sd15,sdxl,sd35",
        help="Deprecated: models are fixed to sd15,sdxl,sd35 and always plotted.",
    )
    parser.add_argument("--rewards-root", type=str, default="")
    parser.add_argument("--geneval-csv", type=str, default="")
    parser.add_argument("--total-steps", type=int, default=-1)
    parser.add_argument("--base-effective-n", type=int, default=2)
    parser.add_argument("--base-k-init", type=int, default=4)
    parser.add_argument("--cutoff-times", type=str, default="16,32")
    parser.add_argument("--base-remaining-particles", type=str, default="2,1")
    parser.add_argument("--effective-ns", type=str, default="1,2,4,8,16")
    parser.add_argument(
        "--sd15-total-steps",
        type=int,
        default=64,
        help="Total diffusion steps T for sd15.",
    )
    parser.add_argument(
        "--sdxl-total-steps",
        type=int,
        default=64,
        help="Total diffusion steps T for sdxl.",
    )
    parser.add_argument(
        "--sd35-total-steps",
        type=int,
        default=32,
        help="Total diffusion steps T for sd35.",
    )
    parser.add_argument(
        "--sd15-cutoff-times",
        type=str,
        default="",
        help='Optional cutoff times for sd15, e.g. "16,32".',
    )
    parser.add_argument(
        "--sdxl-cutoff-times",
        type=str,
        default="",
        help='Optional cutoff times for sdxl, e.g. "16,32".',
    )
    parser.add_argument(
        "--sd35-cutoff-times",
        type=str,
        default="",
        help='Optional cutoff times for sd35, e.g. "8,16".',
    )
    parser.add_argument(
        "--sd15-base-remaining-particles",
        type=str,
        default="",
        help='Optional base ks for sd15, e.g. "2,1".',
    )
    parser.add_argument(
        "--sdxl-base-remaining-particles",
        type=str,
        default="",
        help='Optional base ks for sdxl, e.g. "2,1".',
    )
    parser.add_argument(
        "--sd35-base-remaining-particles",
        type=str,
        default="",
        help='Optional base ks for sd35, e.g. "2,1".',
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["ir", "geneval"],
        default="ir",
        help="Single metric to plot.",
    )
    parser.add_argument(
        "--sd15-flops-json",
        type=str,
        default="paper_figures/figures/scaling/model_step_flops_sd15.json",
        help="Path to sd15 FLOPs JSON.",
    )
    parser.add_argument(
        "--sdxl-flops-json",
        type=str,
        default="paper_figures/figures/scaling/model_step_flops_sdxl.json",
        help="Path to sdxl FLOPs JSON.",
    )
    parser.add_argument(
        "--sd35-flops-json",
        type=str,
        default="paper_figures/figures/scaling/model_step_flops_sd35.json",
        help="Path to sd35 FLOPs JSON.",
    )
    parser.add_argument(
        "--logical-seed",
        type=int,
        default=0,
        help="Logical seed index s. If K samples are needed, uses prompt-local indices [s*K, (s+1)*K-1].",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="paper_figures/figures/scaling/strategies_values",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default="paper_figures/figures/scaling/scaling_strategy.png",
    )
    parser.add_argument(
        "--out-pdf",
        type=str,
        default="paper_figures/figures/scaling/scaling_strategy.pdf",
    )
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, int]:
    model_list = parse_str_list(args.models)
    if len(model_list) != 1:
        raise ValueError("_resolve_paths only supports one model; use _resolve_paths_for_model.")
    cfg = MODEL_CONFIG[model_list[0]]
    rewards_root = Path(args.rewards_root or cfg["rewards_root"]).resolve()
    geneval_csv = Path(args.geneval_csv or cfg["geneval_csv"]).resolve()
    total_steps = int(cfg["total_steps"]) if args.total_steps < 0 else int(args.total_steps)
    return rewards_root, geneval_csv, total_steps


def _resolve_paths_for_model(
    args: argparse.Namespace,
    model: str,
) -> tuple[Path, Path, int]:
    cfg = MODEL_CONFIG[model]
    rewards_root = Path(cfg["rewards_root"]).resolve()
    geneval_csv = Path(cfg["geneval_csv"]).resolve()
    total_steps = int(cfg["total_steps"]) if args.total_steps < 0 else int(args.total_steps)
    return rewards_root, geneval_csv, total_steps


def _load_step_flops(flops_json_path: Path) -> int:
    if not flops_json_path.exists():
        raise FileNotFoundError(f"FLOPs JSON not found: {flops_json_path}")
    payload = json.loads(flops_json_path.read_text(encoding="utf-8"))
    if "one_step_flops" not in payload:
        raise ValueError(f"Missing one_step_flops in {flops_json_path}")
    return int(payload["one_step_flops"])


def _model_schedule_args(
    args: argparse.Namespace,
    model: str,
    default_cutoff_times: list[int],
    default_base_ks: list[int],
) -> tuple[int, list[int], list[int]]:
    if model == "sd15":
        total_steps = int(args.sd15_total_steps)
        cutoff_raw = args.sd15_cutoff_times.strip()
        ks_raw = args.sd15_base_remaining_particles.strip()
    elif model == "sdxl":
        total_steps = int(args.sdxl_total_steps)
        cutoff_raw = args.sdxl_cutoff_times.strip()
        ks_raw = args.sdxl_base_remaining_particles.strip()
    elif model == "sd35":
        total_steps = int(args.sd35_total_steps)
        cutoff_raw = args.sd35_cutoff_times.strip()
        ks_raw = args.sd35_base_remaining_particles.strip()
    else:
        raise ValueError(f"Unknown model: {model}")
    cutoff_times = parse_int_list(cutoff_raw) if cutoff_raw else list(default_cutoff_times)
    base_ks = parse_int_list(ks_raw) if ks_raw else list(default_base_ks)
    if len(cutoff_times) != len(base_ks):
        raise ValueError(
            f"Cutoff/ks length mismatch for {model}: cutoff_times={cutoff_times}, ks={base_ks}"
        )
    return total_steps, cutoff_times, base_ks


def _load_step_metrics_with_local_index(rewards_root: Path) -> pd.DataFrame:
    metric_paths = sorted(rewards_root.glob("*geneval*/metrics.csv"))
    if not metric_paths:
        raise FileNotFoundError(f"No geneval metrics.csv found under {rewards_root}")
    parts = [
        pd.read_csv(p, usecols=["prompt_id", "seed", "step", "image_reward"])
        for p in metric_paths
    ]
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["prompt_id", "seed", "step"], keep="last")
    pair_idx = (
        df[["prompt_id", "seed"]]
        .drop_duplicates()
        .sort_values(["prompt_id", "seed"])
        .reset_index(drop=True)
    )
    pair_idx["local_seed_idx"] = pair_idx.groupby("prompt_id").cumcount()
    return df.merge(pair_idx, on=["prompt_id", "seed"], how="inner")


def _load_geneval(geneval_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(geneval_csv)
    needed = {"prompt_id", "seed", "correct", "tag"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing geneval columns in {geneval_csv}: {sorted(missing)}")
    out = df[["prompt_id", "seed", "correct", "tag"]].copy()
    out["correct"] = out["correct"].astype(float)
    return out


def _scale_strategy(
    *,
    target_n: int,
    base_n: int,
    base_k_init: int,
    cutoff_times: list[int],
    base_remaining_particles: list[int],
) -> Strategy:
    if base_n <= 0:
        raise ValueError("--base-effective-n must be > 0")
    factor = float(target_n) / float(base_n)
    k_init = max(1, int(round(base_k_init * factor)))
    ks = [max(1, int(round(k * factor))) for k in base_remaining_particles]
    # Keep ks valid and non-increasing.
    ks = [min(k_init, ks[0])] + ks[1:]
    for i in range(1, len(ks)):
        ks[i] = min(ks[i], ks[i - 1])
    return Strategy(
        effective_n=int(target_n),
        k_init=int(k_init),
        cutoff_times=[int(t) for t in cutoff_times],
        remaining_particles=[int(k) for k in ks],
    )


def _scheduled_pick_seed(
    prompt_df: pd.DataFrame,
    *,
    strategy: Strategy,
    total_steps: int,
    logical_seed: int,
) -> int:
    # Match baseline scripts: use prompt-local contiguous window.
    # For logical_seed=s and K samples, use indices [s*K, (s+1)*K-1].
    start = logical_seed * strategy.k_init
    end = (logical_seed + 1) * strategy.k_init - 1
    pool = prompt_df[
        (prompt_df["local_seed_idx"] >= start) & (prompt_df["local_seed_idx"] <= end)
    ].copy()
    if pool.empty:
        raise ValueError("No seeds in requested strategy window.")
    survivors = sorted(pool["seed"].astype(int).unique().tolist())
    for cutoff, keep in zip(strategy.cutoff_times, strategy.remaining_particles):
        at_cutoff = pool[(pool["step"] == cutoff) & (pool["seed"].isin(survivors))]
        at_cutoff = at_cutoff.sort_values(["image_reward", "seed"], ascending=[False, True])
        if at_cutoff.empty:
            raise ValueError(f"Missing cutoff rows at step={cutoff}")
        survivors = at_cutoff["seed"].head(keep).astype(int).tolist()
    at_final = pool[(pool["step"] == total_steps) & (pool["seed"].isin(survivors))]
    at_final = at_final.sort_values(["image_reward", "seed"], ascending=[False, True])
    if at_final.empty:
        raise ValueError("Missing final rows for selected survivors.")
    return int(at_final.iloc[0]["seed"])


def _strategy_cache_path(cache_dir: Path, model: str, s: Strategy, logical_seed: int) -> Path:
    cutoff = "-".join(str(v) for v in s.cutoff_times)
    ks = "-".join(str(v) for v in s.remaining_particles)
    name = (
        f"{model}_N{s.effective_n}_kinit{s.k_init}_ts{cutoff}_"
        f"ks{ks}_seed{logical_seed}.json"
    )
    return cache_dir / name


def _bon_cache_path(cache_dir: Path, model: str, effective_n: int, logical_seed: int) -> Path:
    name = f"{model}_bon_N{effective_n}_seed{logical_seed}.json"
    return cache_dir / name


def _best_of_n_pick_seed(prompt_final: pd.DataFrame, *, n: int, logical_seed: int) -> int:
    start = logical_seed * n
    end = (logical_seed + 1) * n - 1
    pool = prompt_final[
        (prompt_final["local_seed_idx"] >= start) & (prompt_final["local_seed_idx"] <= end)
    ].copy()
    if pool.empty:
        raise ValueError("No seeds in requested BoN window.")
    pool = pool.sort_values(["image_reward", "seed"], ascending=[False, True])
    return int(pool.iloc[0]["seed"])


def _geneval_overall_score(selected_geneval: pd.DataFrame) -> float:
    if selected_geneval.empty:
        return float("nan")
    task_scores = selected_geneval.groupby("tag")["correct"].mean()
    return float(task_scores.mean()) if len(task_scores) else float("nan")


def _compute_or_load_strategy_value(
    *,
    strategy: Strategy,
    model: str,
    step_df: pd.DataFrame,
    final_df: pd.DataFrame,
    geneval_df: pd.DataFrame,
    total_steps: int,
    cache_dir: Path,
    logical_seed: int,
) -> dict[str, Any]:
    cache_path = _strategy_cache_path(cache_dir, model, strategy, logical_seed)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    prompt_ids = sorted(step_df["prompt_id"].astype(int).unique().tolist())
    ir_vals: list[float] = []
    selected_pairs: list[tuple[int, int]] = []
    used_prompts: list[int] = []
    for pid in prompt_ids:
        prompt_steps = step_df[step_df["prompt_id"] == pid].copy()
        prompt_final = final_df[final_df["prompt_id"] == pid].copy()
        if prompt_steps.empty or prompt_final.empty:
            continue
        try:
            winner_seed = _scheduled_pick_seed(
                prompt_steps,
                strategy=strategy,
                total_steps=total_steps,
                logical_seed=logical_seed,
            )
        except ValueError:
            continue
        ir_row = prompt_final[prompt_final["seed"] == winner_seed]
        if ir_row.empty:
            continue
        ir_vals.append(float(ir_row.iloc[0]["image_reward"]))
        selected_pairs.append((int(pid), int(winner_seed)))
        used_prompts.append(pid)

    if not ir_vals:
        raise RuntimeError(f"No valid prompt scores found for strategy N={strategy.effective_n}")

    selected_df = pd.DataFrame(selected_pairs, columns=["prompt_id", "seed"])
    selected_geneval = selected_df.merge(geneval_df, on=["prompt_id", "seed"], how="inner")
    payload = {
        "model": model,
        "strategy": "ssp",
        "effective_n": strategy.effective_n,
        "k_init": strategy.k_init,
        "cutoff_times": strategy.cutoff_times,
        "remaining_particles": strategy.remaining_particles,
        "logical_seed": logical_seed,
        "num_prompts": len(ir_vals),
        "mean_ir": float(sum(ir_vals) / len(ir_vals)),
        "mean_geneval": _geneval_overall_score(selected_geneval),
        "prompt_ids": used_prompts,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _compute_or_load_bon_value(
    *,
    effective_n: int,
    model: str,
    final_df: pd.DataFrame,
    geneval_df: pd.DataFrame,
    cache_dir: Path,
    logical_seed: int,
) -> dict[str, Any]:
    cache_path = _bon_cache_path(cache_dir, model, effective_n, logical_seed)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    prompt_ids = sorted(final_df["prompt_id"].astype(int).unique().tolist())
    ir_vals: list[float] = []
    selected_pairs: list[tuple[int, int]] = []
    used_prompts: list[int] = []
    for pid in prompt_ids:
        prompt_final = final_df[final_df["prompt_id"] == pid].copy()
        if prompt_final.empty:
            continue
        try:
            winner_seed = _best_of_n_pick_seed(
                prompt_final,
                n=effective_n,
                logical_seed=logical_seed,
            )
        except ValueError:
            continue
        ir_row = prompt_final[prompt_final["seed"] == winner_seed]
        if ir_row.empty:
            continue
        ir_vals.append(float(ir_row.iloc[0]["image_reward"]))
        selected_pairs.append((int(pid), int(winner_seed)))
        used_prompts.append(pid)

    if not ir_vals:
        raise RuntimeError(f"No valid prompt scores found for BoN N={effective_n}")

    selected_df = pd.DataFrame(selected_pairs, columns=["prompt_id", "seed"])
    selected_geneval = selected_df.merge(geneval_df, on=["prompt_id", "seed"], how="inner")
    payload = {
        "model": model,
        "strategy": "bon",
        "effective_n": int(effective_n),
        "logical_seed": int(logical_seed),
        "num_prompts": len(ir_vals),
        "mean_ir": float(sum(ir_vals) / len(ir_vals)),
        "mean_geneval": _geneval_overall_score(selected_geneval),
        "prompt_ids": used_prompts,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    models = list(PLOT_MODELS)
    if parse_str_list(args.models) != PLOT_MODELS:
        print("Ignoring --models; plotting fixed models: sd15, sdxl, sd35.")
    metric = args.metric.lower()
    if args.rewards_root or args.geneval_csv:
        raise ValueError(
            "--rewards-root/--geneval-csv overrides are not supported in fixed 3-model mode."
        )

    cutoff_times = parse_int_list(args.cutoff_times)
    base_ks = parse_int_list(args.base_remaining_particles)
    effective_ns = parse_int_list(args.effective_ns)
    regret_ns = [2, 4, 8, 16]
    ns_for_compute = sorted(set(effective_ns + regret_ns + [2 * n for n in regret_ns]))
    if len(cutoff_times) != len(base_ks):
        raise ValueError("--cutoff-times and --base-remaining-particles must have same length.")

    flops_paths = {
        "sd15": Path(args.sd15_flops_json).resolve(),
        "sdxl": Path(args.sdxl_flops_json).resolve(),
        "sd35": Path(args.sd35_flops_json).resolve(),
    }
    flops_per_step: dict[str, int] = {
        model: _load_step_flops(path) for model, path in flops_paths.items()
    }

    cache_dir = Path(args.cache_dir).resolve()
    model_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    model_total_steps: dict[str, int] = {}
    for model in models:
        rewards_root, geneval_csv, _ = _resolve_paths_for_model(args, model)
        total_steps, model_cutoff_times, model_base_ks = _model_schedule_args(
            args,
            model,
            cutoff_times,
            base_ks,
        )
        if not rewards_root.exists():
            raise FileNotFoundError(f"Rewards root not found for {model}: {rewards_root}")
        if not geneval_csv.exists():
            raise FileNotFoundError(f"Geneval CSV not found for {model}: {geneval_csv}")
        print(f"Model: {model}")
        print(f"Rewards root: {rewards_root}")
        print(f"Geneval CSV: {geneval_csv}")
        print(
            f"Schedule: T={total_steps}, ts={model_cutoff_times}, ks={model_base_ks}, "
            f"k_init(base)={args.base_k_init}"
        )
        model_total_steps[model] = int(total_steps)

        step_df = _load_step_metrics_with_local_index(rewards_root)
        final_df = step_df[step_df["step"] == total_steps].copy()
        if final_df.empty:
            raise RuntimeError(f"No final-step rows found for {model} at step={total_steps}")
        geneval_df = _load_geneval(geneval_csv)

        records_ssp: list[dict[str, Any]] = []
        records_bon: list[dict[str, Any]] = []
        for target_n in tqdm(ns_for_compute, desc=f"{model} strategies", unit="strategy"):
            if int(target_n) == 1:
                bon_val = _compute_or_load_bon_value(
                    effective_n=1,
                    model=model,
                    final_df=final_df,
                    geneval_df=geneval_df,
                    cache_dir=cache_dir,
                    logical_seed=args.logical_seed,
                )
                # Per request: when N=1 is present, include it only for BoN (not PSP).
                records_bon.append(bon_val)
                continue

            strategy = _scale_strategy(
                target_n=target_n,
                base_n=args.base_effective_n,
                base_k_init=args.base_k_init,
                cutoff_times=model_cutoff_times,
                base_remaining_particles=model_base_ks,
            )
            val = _compute_or_load_strategy_value(
                strategy=strategy,
                model=model,
                step_df=step_df,
                final_df=final_df,
                geneval_df=geneval_df,
                total_steps=total_steps,
                cache_dir=cache_dir,
                logical_seed=args.logical_seed,
            )
            bon_val = _compute_or_load_bon_value(
                effective_n=target_n,
                model=model,
                final_df=final_df,
                geneval_df=geneval_df,
                cache_dir=cache_dir,
                logical_seed=args.logical_seed,
            )
            records_ssp.append(val)
            records_bon.append(bon_val)
        model_records[model] = {
            "ssp": sorted(records_ssp, key=lambda r: int(r["effective_n"])),
            "bon": sorted(records_bon, key=lambda r: int(r["effective_n"])),
        }

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10.6, 4.2))
    ax_n, ax_regret, ax_flops = axes
    plt.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.26, wspace=0.06)
    tick_vals = sorted({int(x) for x in effective_ns})
    for idx, model in enumerate(models):
        color = MODEL_COLORS.get(model, ["#1f77b4", "#2ca02c", "#d4aa00"][idx])
        records_ssp_all = model_records[model]["ssp"]
        records_bon_all = model_records[model]["bon"]
        ssp_ir_by_n = {int(r["effective_n"]): float(r["mean_ir"]) for r in records_ssp_all}
        ssp_ge_by_n = {int(r["effective_n"]): float(r["mean_geneval"]) for r in records_ssp_all}
        bon_ir_by_n = {int(r["effective_n"]): float(r["mean_ir"]) for r in records_bon_all}
        bon_ge_by_n = {int(r["effective_n"]): float(r["mean_geneval"]) for r in records_bon_all}

        xs_ssp = [n for n in effective_ns if n in ssp_ir_by_n]
        xs_bon = [n for n in effective_ns if n in bon_ir_by_n]
        ys_ir_ssp = [ssp_ir_by_n[n] for n in xs_ssp]
        ys_ge_ssp = [ssp_ge_by_n[n] for n in xs_ssp]
        ys_ir_bon = [bon_ir_by_n[n] for n in xs_bon]
        ys_ge_bon = [bon_ge_by_n[n] for n in xs_bon]
        t_steps = int(model_total_steps[model])
        step_flops = int(flops_per_step[model])
        x_flops_ssp = [int(n) * step_flops * t_steps for n in xs_ssp]
        x_flops_bon = [int(n) * step_flops * t_steps for n in xs_bon]
        regret_x: list[int] = []
        regret_y: list[float] = []
        for n in regret_ns:
            if metric == "ir":
                if (n in ssp_ir_by_n) and ((2 * n) in bon_ir_by_n):
                    regret_x.append(n)
                    regret_y.append(bon_ir_by_n[2 * n] - ssp_ir_by_n[n])
            else:
                if (n in ssp_ge_by_n) and ((2 * n) in bon_ge_by_n):
                    regret_x.append(n)
                    regret_y.append(bon_ge_by_n[2 * n] - ssp_ge_by_n[n])

        print(f"Model: {model}")
        print(f"FLOPs/step: {step_flops}")
        print(f"T: {t_steps}")
        print("Per effective N totals:")
        for n_i, prod in zip(xs_bon, x_flops_bon):
            print(f"  effective_n={n_i}")
            print(f"  total_flops={n_i} * {step_flops} * {t_steps} = {prod}")
        print("")

        if metric == "ir":
            y_ssp = ys_ir_ssp
            y_bon = ys_ir_bon
        else:
            y_ssp = ys_ge_ssp
            y_bon = ys_ge_bon

        ax_n.plot(
            xs_bon,
            y_bon,
            marker="o",
            linewidth=2.0,
            linestyle="--",
            color=color,
            label=f"{model} BoN",
            zorder=2,
        )
        ax_n.plot(
            xs_ssp,
            y_ssp,
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"{model} PSP",
            zorder=3,
        )
        ax_flops.plot(
            x_flops_bon,
            y_bon,
            marker="o",
            linewidth=2.0,
            linestyle="--",
            color=color,
            label=f"{model} BoN",
            zorder=2,
        )
        ax_flops.plot(
            x_flops_ssp,
            y_ssp,
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"{model} PSP",
            zorder=3,
        )
        ax_regret.plot(
            regret_x,
            regret_y,
            marker="o",
            linewidth=2.2,
            linestyle="-",
            color=color,
            label=f"{model} regret",
            zorder=3,
        )

    ax_n.set_xscale("log", base=2)
    ax_n.set_xticks(tick_vals)
    ax_n.set_xticklabels([str(v) for v in tick_vals])
    ax_n.set_xlabel(r"Effective $\bar{N}$")
    if metric == "ir":
        ax_n.set_ylabel("IR")
        ax_n.set_title(r"IR vs Effective $\bar{N}$")
    else:
        ax_n.set_ylabel("GenEval")
        ax_n.set_title("GenEval vs Effective N")
    ax_n.grid(True, axis="y", alpha=0.25)
    ax_n.grid(True, axis="x", alpha=0.25)
    ax_n.set_box_aspect(1)

    ax_regret.set_xscale("log", base=2)
    regret_ticks = [n for n in regret_ns if n in ns_for_compute]
    ax_regret.set_xticks(regret_ticks)
    ax_regret.set_xticklabels([str(v) for v in regret_ticks])
    ax_regret.set_xlabel(r"Effective $\bar{N}$")
    if metric == "ir":
        ax_regret.set_ylabel("Regret (IR)")
    else:
        ax_regret.set_ylabel("Regret (GenEval)")
    ax_regret.set_title(r"Regret: BoN($2\bar{N}$) - PSP($\bar{N}$)")
    ax_regret.grid(True, axis="y", alpha=0.25)
    ax_regret.grid(True, axis="x", alpha=0.25)
    ax_regret.set_ylim(bottom=0.0)
    ax_regret.set_box_aspect(1)

    ax_flops.set_xscale("log")
    ax_flops.set_xlabel("Total Generation FLOPs")
    if metric == "ir":
        ax_flops.set_ylabel("IR")
        ax_flops.set_title("IR vs Total Generation FLOPs")
    else:
        ax_flops.set_ylabel("GenEval")
        ax_flops.set_title("GenEval vs Total Generation FLOPs")
    ax_flops.grid(True, axis="y", alpha=0.25)
    ax_flops.grid(True, axis="x", alpha=0.25)
    ax_flops.set_box_aspect(1)

    legend_model_labels = {"sd15": "SD v1.5", "sdxl": "SDXL", "sd35": "SD 3.5"}
    legend_handles: list[Line2D] = [
        Line2D([], [], linestyle="None", label=r"$\bf{Line\ Color:}$"),
    ]
    for idx, model in enumerate(models):
        color = MODEL_COLORS.get(model, ["#1f77b4", "#2ca02c", "#d4aa00"][idx])
        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=2.5, label=legend_model_labels.get(model, model))
        )
    legend_handles.append(Line2D([], [], linestyle="None", label=r"$\bf{Line\ Style:}$"))
    legend_handles.append(Line2D([0], [0], color="black", linewidth=2.5, linestyle="-", label="PSP"))
    legend_handles.append(Line2D([0], [0], color="black", linewidth=2.5, linestyle="--", label="BoN"))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=len(legend_handles),
        frameon=False,
    )

    out_png = Path(args.out_png).resolve()
    out_pdf = Path(args.out_pdf).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=350, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=350, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Strategy cache dir: {cache_dir}")
    for model in models:
        for r in model_records[model]["ssp"]:
            print(
                "{model} PSP N={N:>3d} | k_init={k_init:>3d} | ks={ks} | IR={ir:.4f} | GenEval={ge:.4f} | prompts={np}".format(
                    model=model,
                    N=int(r["effective_n"]),
                    k_init=int(r.get("k_init", -1)),
                    ks=r.get("remaining_particles", []),
                    ir=float(r["mean_ir"]),
                    ge=float(r["mean_geneval"]),
                    np=int(r["num_prompts"]),
                )
            )
        for r in model_records[model]["bon"]:
            print(
                "{model} BoN N={N:>3d} | IR={ir:.4f} | GenEval={ge:.4f} | prompts={np}".format(
                    model=model,
                    N=int(r["effective_n"]),
                    ir=float(r["mean_ir"]),
                    ge=float(r["mean_geneval"]),
                    np=int(r["num_prompts"]),
                )
            )


if __name__ == "__main__":
    main()
