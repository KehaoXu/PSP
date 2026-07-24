#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image


MODEL_CONFIG: dict[str, dict[str, Any]] = {
    "sdxl": {
        "name": "Stable Diffusion XL",
        "rewards_root": "results/reward_signal/sdxl_reward_signal",
        "fk_root": "results/fk_steering/sdxl/fk_sdxl_k4_t64_eta1_ir_geneval",
        "total_steps": 64,
        "cutoff_times": [16, 32],
    },
    "sd15": {
        "name": "Stable Diffusion v1.5",
        "rewards_root": "results/reward_signal/sd15_reward_signal",
        "fk_root": "results/fk_steering/sd15/fk_sdv15_k4_t64_eta1_ir_geneval",
        "total_steps": 64,
        "cutoff_times": [16, 32],
    },
    "sd35": {
        "name": "Stable Diffusion 3.5",
        "rewards_root": "results/reward_signal/sd35_reward_signal",
        "fk_root": "results/fk_steering/sd35/fk_sdv35_k4_t32_gamma_0p005_ir_geneval",
        "total_steps": 32,
        "cutoff_times": [8, 16],
    },
}


def parse_int_list(value: str) -> list[int]:
    vals = [x.strip() for x in value.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return [int(v) for v in vals]


def parse_dot_jitter(value: str) -> dict[str, dict[int, float]]:
    """
    Parse jitter spec entries like:
      "pps:25:0.08,bon:75:0.04"
    where each token is "<algorithm>:<percent>:<amplitude>".
    """
    out: dict[str, dict[int, float]] = {}
    if not value.strip():
        return out
    allowed = {"bon", "fk", "pps"}
    tokens = [t.strip() for t in value.split(",") if t.strip()]
    for tok in tokens:
        parts = [p.strip() for p in tok.split(":")]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --dot-jitter token '{tok}'. Expected format <algorithm>:<percent>:<amplitude>."
            )
        algo = parts[0].lower()
        if algo not in allowed:
            raise ValueError(f"Unknown jitter algorithm '{algo}'. Allowed: bon,fk,pps.")
        pct = int(round(float(parts[1])))
        amp = float(parts[2])
        out.setdefault(algo, {})[pct] = amp
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create methods trajectory figure with 3 panels: Best-of-N, "
            "Importance Sampling (FK final-only), Progressive Seed Pruning."
        )
    )
    parser.add_argument("--prompt-id", type=int, default=220)
    parser.add_argument("--model", type=str, choices=sorted(MODEL_CONFIG.keys()), default="sdxl")
    parser.add_argument("--rewards-root", type=str, default="")
    parser.add_argument("--fk-root", type=str, default="")
    parser.add_argument("--fk-seed-id", type=int, default=42)
    parser.add_argument("--fk-seed-dir", type=str, default="")
    parser.add_argument(
        "--no-fk-intermediates",
        action="store_true",
        help="Use precomputed FK final results only (skip live FK trajectory run).",
    )
    parser.add_argument(
        "--force-fk-rerun",
        action="store_true",
        help="Ignore cached FK intermediates and rerun FK steering trajectory collection.",
    )
    parser.add_argument("--k-bon", type=int, default=4)
    parser.add_argument("--k-init", type=int, default=8)
    parser.add_argument("--cutoff-times", type=str, default="")
    parser.add_argument("--remaining-particles", type=str, default="4,2")
    parser.add_argument("--total-steps", type=int, default=-1)
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default="",
    )
    parser.add_argument(
        "--out-pdf",
        type=str,
        default="",
    )
    parser.add_argument("--dpi", type=int, default=350)
    parser.add_argument("--thumb-zoom", type=float, default=0.045)
    parser.add_argument(
        "--thumb-x-offset",
        type=float,
        default=2.8,
        help="Horizontal offset (in x-axis units) from final 100% point to thumbnail anchor.",
    )
    parser.add_argument(
        "--repeat-xticks",
        action="store_true",
        help="Show x-tick labels on both trajectory (top) and compute (bottom) rows.",
    )
    parser.add_argument(
        "--dot-jitter",
        type=str,
        default="",
        help="Comma-separated jitter specs: <algorithm>:<percent>:<amplitude> for Y-axis only (e.g., 'pps:25:0.08').",
    )
    parser.add_argument(
        "--fk-bars-continue-first-child",
        action="store_true",
        help="FK compute bars: continue a parent as its first child. If not set, bars break at every checkpoint.",
    )
    return parser.parse_args()


def _read_metadata_prompt(metadata_path: Path, prompt_id: int) -> str:
    rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if prompt_id < 0 or prompt_id >= len(rows):
        raise ValueError(f"prompt-id out of range for metadata file: {prompt_id}")
    return str(rows[prompt_id].get("prompt", ""))


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, int, list[int]]:
    cfg = MODEL_CONFIG[args.model]
    rewards_root = Path(args.rewards_root or cfg["rewards_root"]).resolve()
    fk_root = Path(args.fk_root or cfg["fk_root"]).resolve()
    total_steps = int(cfg["total_steps"]) if args.total_steps < 0 else int(args.total_steps)
    cutoff_times = parse_int_list(args.cutoff_times) if args.cutoff_times else list(cfg["cutoff_times"])
    return rewards_root, fk_root, total_steps, cutoff_times


def _load_prompt_traj(rewards_root: Path, prompt_id: int) -> pd.DataFrame:
    metric_paths = sorted(rewards_root.glob("*geneval*/metrics.csv"))
    if not metric_paths:
        raise FileNotFoundError(f"No geneval metrics.csv found under {rewards_root}")
    parts = []
    for p in metric_paths:
        df = pd.read_csv(p, usecols=["prompt_id", "seed", "step", "image_reward"])
        df = df[df["prompt_id"] == prompt_id].copy()
        if not df.empty:
            parts.append(df)
    if not parts:
        raise ValueError(f"No trajectory rows found for prompt_id={prompt_id} under {rewards_root}")
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["prompt_id", "seed", "step"], keep="last")
    pair = (
        out[["prompt_id", "seed"]]
        .drop_duplicates()
        .sort_values(["prompt_id", "seed"])
        .reset_index(drop=True)
    )
    pair["local_seed_idx"] = pair.groupby("prompt_id").cumcount()
    return out.merge(pair, on=["prompt_id", "seed"], how="inner")


def _sample_path_map_for_prompt(rewards_root: Path, prompt_id: int) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    prompt_name = f"{prompt_id:05d}"
    for samples_root in sorted(rewards_root.glob("*geneval*/samples")):
        prompt_dir = samples_root / prompt_name
        if not prompt_dir.exists():
            continue
        for png in sorted(prompt_dir.glob("*.png")):
            if png.stem.isdigit():
                mapping[int(png.stem)] = png
    return mapping


def _resolve_fk_seed_dir(fk_root: Path, requested: str, seed_id: int) -> Path:
    if requested:
        p = fk_root / requested
        if not p.exists():
            raise FileNotFoundError(f"--fk-seed-dir not found: {p}")
        return p
    preferred = sorted(p for p in fk_root.glob(f"seed={seed_id}_*") if p.is_dir())
    if preferred:
        return preferred[0]
    all_seeds = sorted(p for p in fk_root.glob("seed=*_*") if p.is_dir())
    if not all_seeds:
        raise FileNotFoundError(f"No seed dirs found in FK root: {fk_root}")
    return all_seeds[0]


def _round_to_nearest_ten(value: float) -> int:
    return int(10 * round(float(value) / 10.0))


def _collect_fk_trajectory_live(
    *,
    repo_root: Path,
    fk_seed_dir: Path,
    prompt_id: int,
    prompt_text: str,
    seed_id: int,
    force_rerun: bool = False,
) -> dict[str, Any]:
    fk_prompt_dir = fk_seed_dir / f"{prompt_id:05d}"
    cache_json = fk_prompt_dir / f"fk_live_trajectory_seed{seed_id}.json"
    cache_samples_dir = fk_prompt_dir / f"fk_live_samples_seed{seed_id}"

    args_path = fk_seed_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing FK args.json at {args_path}")
    run_args = json.loads(args_path.read_text(encoding="utf-8"))

    text_to_image_dir = repo_root / "Fk-Diffusion-Steering" / "text_to_image"
    if str(text_to_image_dir) not in sys.path:
        sys.path.insert(0, str(text_to_image_dir))
    # FK modules use absolute imports like `from smc_utils import ...`,
    # so add the fkd_diffusers package directory explicitly as well.
    fkd_diffusers_dir = text_to_image_dir / "fkd_diffusers"
    if str(fkd_diffusers_dir) not in sys.path:
        sys.path.insert(0, str(fkd_diffusers_dir))

    import torch
    from diffusers import DDIMScheduler, UNet2DConditionModel
    from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion
    from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL
    from fkd_diffusers.rewards import get_reward_function

    model_name = str(run_args.get("model_name", "stabilityai/stable-diffusion-xl-base-1.0"))
    num_particles = int(run_args.get("num_particles", 4))
    num_inference_steps = int(run_args.get("num_inference_steps", 64))
    guidance_reward_fn = str(run_args.get("guidance_reward_fn", "ImageReward"))
    lmbda = float(run_args.get("lmbda", 10.0))
    eta = float(run_args.get("eta", 1.0))
    adaptive_resampling = bool(run_args.get("adaptive_resampling", False))
    resample_frequency = int(run_args.get("resample_frequency", 12))
    resample_t_start = int(run_args.get("resample_t_start", 12))
    resample_t_end = int(run_args.get("resample_t_end", max(1, num_inference_steps - 1)))
    potential_type = str(run_args.get("potential_type", "max"))

    if cache_json.exists() and not force_rerun:
        cached = json.loads(cache_json.read_text(encoding="utf-8"))
        sample_paths = [
            str(p)
            for p in sorted(cache_samples_dir.glob("*.png"))
            if p.stem.isdigit()
        ]
        cache_compatible = (
            int(cached.get("time_steps", -1)) == num_inference_steps
            and int(cached.get("num_particles", -1)) == num_particles
            and int(cached.get("resample_frequency", -1)) == resample_frequency
            and int(cached.get("resampling_t_start", -1)) == resample_t_start
            and int(cached.get("resampling_t_end", -1)) == resample_t_end
            and 0 in [int(v) for v in cached.get("timesteps", [])]
            and len(sample_paths) >= num_particles
        )
        if cache_compatible:
            cached["sample_paths"] = sample_paths
            return cached

    torch.manual_seed(seed_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_id)
        torch.cuda.manual_seed_all(seed_id)

    if "xl" in model_name and "dpo" not in model_name:
        pipe = FKDStableDiffusionXL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        )
    elif "mhdang/dpo" in model_name and "xl" in model_name:
        pipe = FKDStableDiffusionXL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        unet_id = "mhdang/dpo-sdxl-text2image-v1"
        unet = UNet2DConditionModel.from_pretrained(
            unet_id, subfolder="unet", torch_dtype=torch.float16
        )
        pipe.unet = unet
    elif "mhdang/dpo" in model_name and "xl" not in model_name:
        pipe = FKDStableDiffusion.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        unet_id = "mhdang/dpo-sd1.5-text2image-v1"
        unet = UNet2DConditionModel.from_pretrained(
            unet_id, subfolder="unet", torch_dtype=torch.float16
        )
        pipe.unet = unet
    else:
        pipe = FKDStableDiffusion.from_pretrained(model_name, torch_dtype=torch.float16)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    fkd_args = dict(
        lmbda=lmbda,
        num_particles=num_particles,
        use_smc=True,
        adaptive_resampling=adaptive_resampling,
        resample_frequency=resample_frequency,
        time_steps=num_inference_steps,
        resampling_t_start=resample_t_start,
        resampling_t_end=resample_t_end,
        guidance_reward_fn=guidance_reward_fn,
        potential_type=potential_type,
        log_reward_every_step=True,
        # For this figure, do not force FK resampling at final step (T-1).
        disable_forced_final_resampling=True,
    )
    prompts = [prompt_text] * num_particles
    output = pipe(
        prompts,
        num_inference_steps=num_inference_steps,
        eta=eta,
        fkd_args=fkd_args,
    )
    trajectory = getattr(output, "trajectory_log_raw", None) or getattr(output, "trajectory_log", None)
    if trajectory is None:
        raise RuntimeError("FK run did not return trajectory log.")

    cache_samples_dir.mkdir(parents=True, exist_ok=True)
    sample_paths: list[str] = []
    for idx, im in enumerate(output.images):
        img_path = cache_samples_dir / f"{idx:05d}.png"
        im.save(img_path)
        sample_paths.append(str(img_path))
    final_image_rewards_raw = get_reward_function(
        reward_name=guidance_reward_fn,
        images=[im for im in output.images],
        prompts=prompts,
    )
    final_image_rewards = [float(v) for v in final_image_rewards_raw]

    resampling_steps = list(
        np.arange(resample_t_start, resample_t_end + 1, resample_frequency).astype(int).tolist()
    )
    if (num_inference_steps - 1) not in resampling_steps:
        resampling_steps.append(num_inference_steps - 1)
    resampling_steps = sorted(set(resampling_steps))

    payload = {
        "timesteps": trajectory.get("timesteps", []),
        "rewards": trajectory.get("rewards", []),
        "kills": trajectory.get("kills", []),
        "parents": trajectory.get("parents", []),
        "num_particles": int(trajectory.get("num_particles", num_particles)),
        "time_steps": int(trajectory.get("time_steps", num_inference_steps)),
        "resampling_t_start": int(trajectory.get("resampling_t_start", resample_t_start)),
        "resampling_t_end": int(trajectory.get("resampling_t_end", resample_t_end)),
        "resample_frequency": int(trajectory.get("resample_frequency", resample_frequency)),
        "resampling_steps": resampling_steps,
        "final_image_rewards": final_image_rewards,
    }
    cache_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["sample_paths"] = sample_paths
    return payload


def _load_fk_final_only(
    *,
    fk_prompt_dir: Path,
    fk_seed_dir: Path,
) -> dict[str, Any]:
    results_path = fk_prompt_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing FK results.json: {results_path}")
    data = json.loads(results_path.read_text(encoding="utf-8"))
    vals = data.get("ImageReward", {}).get("result", [])
    if not isinstance(vals, list) or len(vals) == 0:
        raise ValueError(f"Invalid ImageReward.result in {results_path}")

    sample_paths = [
        str(p) for p in sorted((fk_prompt_dir / "samples").glob("*.png")) if p.stem.isdigit()
    ]
    if len(sample_paths) < len(vals):
        sample_paths = [
            str(p)
            for p in sorted((fk_prompt_dir / "best_of_n_samples").glob("*.png"))
            if p.stem.isdigit()
        ]

    args_path = fk_seed_dir / "args.json"
    run_args = json.loads(args_path.read_text(encoding="utf-8")) if args_path.exists() else {}
    num_steps = int(run_args.get("num_inference_steps", 64))
    resample_frequency = int(run_args.get("resample_frequency", 12))
    resample_t_start = int(run_args.get("resample_t_start", 12))
    resample_t_end = int(run_args.get("resample_t_end", max(1, num_steps - 1)))
    resampling_steps = list(np.arange(resample_t_start, resample_t_end + 1, resample_frequency).astype(int))
    if (num_steps - 1) not in resampling_steps:
        resampling_steps.append(num_steps - 1)
    resampling_steps = sorted(set(int(v) for v in resampling_steps))

    # Final-only trajectory, equivalent to old behavior.
    return {
        "timesteps": [num_steps - 1],
        "rewards": [[float(v) for v in vals]],
        "final_image_rewards": [float(v) for v in vals],
        "parents": [[i for i in range(len(vals))]],
        "kills": [[0 for _ in range(len(vals))]],
        "num_particles": len(vals),
        "time_steps": num_steps,
        "resampling_t_start": resample_t_start,
        "resampling_t_end": resample_t_end,
        "resample_frequency": resample_frequency,
        "resampling_steps": resampling_steps,
        "sample_paths": sample_paths,
    }


def _load_thumb_array(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _add_thumb(
    ax: plt.Axes,
    img: np.ndarray,
    x: float,
    y: float,
    zoom: float,
    *,
    edgecolor: str = "black",
    linewidth: float = 1.0,
) -> None:
    ab = AnnotationBbox(
        OffsetImage(img, zoom=zoom),
        (x, y),
        frameon=True,
        pad=0.2,
        box_alignment=(0.0, 0.5),
        bboxprops=dict(edgecolor=edgecolor, linewidth=linewidth),
    )
    ax.add_artist(ab)


def _style_panel(
    ax: plt.Axes,
    title: str,
    x_positions: list[float],
    x_labels: list[str],
    y_label: bool = False,
    *,
    show_xlabel: bool = True,
    show_xticklabels: bool = True,
) -> None:
    ax.set_title(title, fontsize=14, pad=10)
    for x in x_positions:
        ax.axvline(x, color="#cccccc", linewidth=1.0, zorder=0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels if show_xticklabels else [], fontsize=11)
    ax.tick_params(axis="x", labelbottom=show_xticklabels)
    if show_xlabel:
        ax.set_xlabel("inference progress (%)", fontsize=11, labelpad=6)
    else:
        ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=10)
    if y_label:
        ax.set_ylabel("Reward (IR)", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)


def _style_compute_panel(ax: plt.Axes, x_positions: list[float], x_labels: list[str]) -> None:
    for x in x_positions:
        ax.axvline(x, color="#cccccc", linewidth=1.0, zorder=0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel("inference progress (%)", fontsize=11, labelpad=6)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)


def _draw_arrow_path(ax: plt.Axes, xs: list[float], ys: list[float], *, color: str = "#666666", lw: float = 1.4) -> None:
    if len(xs) < 2:
        return
    for i in range(len(xs) - 1):
        ax.annotate(
            "",
            xy=(xs[i + 1], ys[i + 1]),
            xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=0),
            zorder=2,
        )


def _progress_labels(stage_steps: list[int], total_steps: int) -> list[str]:
    if total_steps <= 1:
        return ["0"] + ["100" for _ in stage_steps[1:]]
    labels: list[str] = []
    for i, s in enumerate(stage_steps):
        if i == 0:
            pct = 0
        elif i == len(stage_steps) - 1:
            pct = 100
        else:
            pct = int(round(100.0 * float(s) / float(total_steps)))
            pct = max(0, min(100, pct))
        labels.append(f"{pct}")
    return labels


def _progress_positions(stage_steps: list[int], total_steps: int) -> list[float]:
    if total_steps <= 1:
        return [0.0] + [100.0 for _ in stage_steps[1:]]
    pos: list[float] = []
    for i, s in enumerate(stage_steps):
        if i == 0:
            p = 0.0
        elif i == len(stage_steps) - 1:
            p = 100.0
        else:
            p = 100.0 * float(s) / float(total_steps)
            p = max(0.0, min(100.0, p))
        pos.append(p)
    return pos


def _jitter_offsets(n: int, amplitude: float) -> list[float]:
    if n <= 1 or amplitude == 0:
        return [0.0 for _ in range(n)]
    return np.linspace(-amplitude, amplitude, n).tolist()


def _dot_y_with_jitter(
    *,
    x: float,
    y: float,
    algorithm: str,
    seed_idx: int,
    n_seeds: int,
    jitter_map: dict[str, dict[int, float]],
) -> float:
    pct = int(round(float(x)))
    amp = float(jitter_map.get(algorithm, {}).get(pct, 0.0))
    if amp == 0.0:
        return y
    offsets = _jitter_offsets(n_seeds, amp)
    idx = min(max(seed_idx, 0), max(0, n_seeds - 1))
    return y + offsets[idx]


def _stack_gap_from_bounds(y_bounds: tuple[float, float] | None, y_fallback: tuple[float, float]) -> float:
    low, high = y_bounds if y_bounds is not None else y_fallback
    span = max(1e-6, float(high) - float(low))
    # Slightly larger fixed spacing to keep thumbnails clearly separated.
    return 0.20 * span


def _stack_thumbnail_y_positions(
    *,
    top_y: float,
    n_items: int,
    gap: float,
    y_bounds: tuple[float, float] | None,
    y_fallback: tuple[float, float],
) -> list[float]:
    if n_items <= 0:
        return []
    ys = [top_y - i * gap for i in range(n_items)]
    low, high = y_bounds if y_bounds is not None else y_fallback
    span = max(1e-6, float(high) - float(low))
    # Keep lowest thumbnail above panel baseline (x-axis area).
    min_allowed = float(low) + 0.04 * span
    if ys[-1] < min_allowed:
        shift = min_allowed - ys[-1]
        ys = [y + shift for y in ys]
    return ys


def _plot_bon_panel(
    ax: plt.Axes,
    prompt_df: pd.DataFrame,
    sample_map: dict[int, Path],
    *,
    k_bon: int,
    total_steps: int,
    cutoff_times: list[int],
    thumb_zoom: float,
    thumb_x_offset: float,
    y_bounds: tuple[float, float] | None = None,
    jitter_map: dict[str, dict[int, float]] | None = None,
    repeat_xticks: bool = False,
) -> None:
    jitter_map = jitter_map or {}
    extra_step = int(round(0.75 * total_steps))
    stage_steps = [1] + cutoff_times + [total_steps]
    if 1 < extra_step < total_steps:
        stage_steps.append(extra_step)
    stage_steps = sorted(set(stage_steps))
    x = _progress_positions(stage_steps, total_steps)
    xlabels = _progress_labels(stage_steps, total_steps)
    _style_panel(
        ax,
        "Best-of-N",
        x,
        xlabels,
        y_label=True,
        show_xlabel=False,
        show_xticklabels=repeat_xticks,
    )

    pool = prompt_df[(prompt_df["local_seed_idx"] >= 0) & (prompt_df["local_seed_idx"] < k_bon)].copy()
    if pool.empty:
        raise ValueError("No Best-of-N candidates found for prompt.")
    seed_ids = sorted(pool["seed"].astype(int).unique().tolist())
    final_rows = pool[pool["step"] == total_steps].sort_values(["image_reward", "seed"], ascending=[False, True])
    if final_rows.empty:
        raise ValueError("No final-step rows for Best-of-N pool.")
    winner_seed = int(final_rows.iloc[0]["seed"])
    y_min = float(pool["image_reward"].min())
    y_max = float(pool["image_reward"].max())
    y_pad = 0.08 * max(1e-6, y_max - y_min)

    for seed_rank, seed in enumerate(seed_ids):
        one = pool[pool["seed"] == seed]
        ys: list[float] = []
        xj: list[float] = []
        for s in stage_steps:
            row = one[one["step"] == s]
            ys.append(float(row.iloc[0]["image_reward"]) if not row.empty else np.nan)
        yj = [
            _dot_y_with_jitter(
                x=float(x[i]),
                y=float(ys[i]),
                algorithm="bon",
                seed_idx=seed_rank,
                n_seeds=len(seed_ids),
                jitter_map=jitter_map,
            )
            for i in range(len(ys))
        ]
        _draw_arrow_path(ax, [float(v) for v in x], yj, color="#666666", lw=1.4)
        # Intermediate points are black dots.
        if len(x) > 2:
            ax.scatter(x[1:-1], yj[1:-1], s=22, color="#1f77b4", edgecolors="black", linewidths=0.8, zorder=4)
        # First point (t=T) always black.
        ax.scatter([x[0]], [yj[0]], s=22, color="#1f77b4", edgecolors="black", linewidths=0.8, zorder=4)
        if seed != winner_seed:
            # Best-of-N keeps all samples until final selection.
            ax.scatter([x[-1]], [yj[-1]], s=24, color="#1f77b4", edgecolors="black", linewidths=0.8, zorder=5)
        else:
            # Winner final point highlighted in green.
            ax.scatter([x[-1]], [yj[-1]], s=28, color="#2ca02c", edgecolors="black", linewidths=0.9, zorder=6)

    # Show thumbnails for all final Best-of-N candidates (ordered by final reward).
    thumb_gap = _stack_gap_from_bounds(y_bounds, (y_min, y_max))
    top_final_y = float(final_rows.iloc[0]["image_reward"])
    stack_ys = _stack_thumbnail_y_positions(
        top_y=top_final_y,
        n_items=len(final_rows),
        gap=thumb_gap,
        y_bounds=y_bounds,
        y_fallback=(y_min, y_max),
    )
    for rank, row in enumerate(final_rows.itertuples(index=False)):
        seed = int(row.seed)
        img_path = sample_map.get(seed)
        if img_path is None:
            continue
        y_thumb = stack_ys[rank]
        _add_thumb(
            ax,
            _load_thumb_array(img_path),
            x[-1] + thumb_x_offset,
            y_thumb,
            thumb_zoom,
            edgecolor="#2ca02c" if rank == 0 else "black",
            linewidth=1.6 if rank == 0 else 1.0,
        )

    ax.set_xlim(min(x) - 2.0, x[-1] + thumb_x_offset + 1.0)
    if y_bounds is None:
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
    else:
        ax.set_ylim(y_bounds[0], y_bounds[1])


def _plot_fk_panel(
    ax: plt.Axes,
    fk_traj: dict[str, Any],
    *,
    thumb_zoom: float,
    thumb_x_offset: float,
    y_bounds: tuple[float, float] | None = None,
    jitter_map: dict[str, dict[int, float]] | None = None,
    repeat_xticks: bool = False,
) -> None:
    jitter_map = jitter_map or {}
    timesteps_all = [int(v) for v in fk_traj.get("timesteps", [])]
    rewards_all = [[float(x) for x in row] for row in fk_traj.get("rewards", [])]
    parents_all = [[int(x) for x in row] for row in fk_traj.get("parents", [])]
    n_particles = int(fk_traj.get("num_particles", 4))
    total_steps = int(fk_traj.get("time_steps", 64))
    if not timesteps_all or not rewards_all:
        raise ValueError("FK trajectory is empty.")
    if len(timesteps_all) != len(rewards_all):
        raise ValueError("FK trajectory timesteps/rewards length mismatch.")
    if len(parents_all) != len(timesteps_all):
        # Backward compatibility if parents are missing.
        parents_all = [[i for i in range(n_particles)] for _ in timesteps_all]

    # Place x ticks at configured resampling steps, rounded to nearest multiple of 5.
    raw_resampling_steps = [int(v) for v in fk_traj.get("resampling_steps", [])]
    if not raw_resampling_steps:
        raw_resampling_steps = [int(v) for v in timesteps_all]
    # Show only t=0 + resampling checkpoints + final step in FK panel.
    selected_steps = sorted(set([0] + raw_resampling_steps + [max(timesteps_all)]))
    idx_by_step = {t: i for i, t in enumerate(timesteps_all)}
    selected_indices = [idx_by_step[s] for s in selected_steps if s in idx_by_step]
    timesteps = [timesteps_all[i] for i in selected_indices]
    rewards = [rewards_all[i] for i in selected_indices]
    if not timesteps:
        # Final-only fallback.
        timesteps = [max(timesteps_all)]
        rewards = [rewards_all[-1]]
        selected_indices = [len(timesteps_all) - 1]
    # Ensure final FK dots align with actually produced final images.
    final_image_rewards = [float(v) for v in fk_traj.get("final_image_rewards", [])]
    if final_image_rewards and len(final_image_rewards) == len(rewards[-1]):
        rewards[-1] = final_image_rewards

    tick_steps = sorted(set([0] + raw_resampling_steps + [max(timesteps)]))
    tick_pairs: list[tuple[float, str]] = []
    for st in tick_steps:
        if total_steps <= 1:
            pct = 100.0
        else:
            pct = 100.0 * float(st) / float(total_steps - 1)
        rounded = float(_round_to_nearest_ten(pct))
        tick_pairs.append((rounded, str(int(rounded))))
    # Place ticks/gridlines at rounded label positions (as requested).
    # Keep first occurrence if multiple steps round to same decile.
    dedup_ticks: dict[float, str] = {}
    for pos, label in tick_pairs:
        if pos not in dedup_ticks:
            dedup_ticks[pos] = label
    tick_positions = sorted(dedup_ticks.keys())
    tick_labels = [dedup_ticks[p] for p in tick_positions]
    _style_panel(
        ax,
        "Importance Sampling",
        tick_positions,
        tick_labels,
        y_label=False,
        show_xlabel=False,
        show_xticklabels=repeat_xticks,
    )

    x_positions = []
    for st in timesteps:
        if total_steps <= 1:
            pct = 100.0
        else:
            pct = 100.0 * float(st) / float(max(1, total_steps - 1))
        x_positions.append(float(_round_to_nearest_ten(pct)))

    # Precompute killed indices per checkpoint so killed points are shown as x only.
    killed_by_step: dict[int, set[int]] = {}
    for step_idx in range(1, len(timesteps)):
        prev_y = rewards[step_idx - 1]
        cur_y = rewards[step_idx]
        # Parent mapping belongs to the PREVIOUS selected checkpoint's resampling event.
        prev_full_idx = selected_indices[step_idx - 1]
        cur_parents = (
            parents_all[prev_full_idx]
            if prev_full_idx < len(parents_all)
            else list(range(n_particles))
        )
        parent_set = set(cur_parents[: len(cur_y)])
        killed = {i for i in range(len(prev_y)) if i not in parent_set}
        if timesteps[step_idx - 1] == 0:
            killed = set()
        killed_by_step[step_idx - 1] = killed
    killed_by_step[len(timesteps) - 1] = set()

    # Precompute jittered y-values per checkpoint/particle.
    yj_by_step: list[list[float]] = []
    for step_i, (xv, ys) in enumerate(zip(x_positions, rewards)):
        yj_by_step.append(
            [
                _dot_y_with_jitter(
                    x=xv,
                    y=float(ys[i]),
                    algorithm="fk",
                    seed_idx=i,
                    n_seeds=len(ys),
                    jitter_map=jitter_map,
                )
                for i in range(len(ys))
            ]
        )

    # Draw checkpoint dots/x markers: no blue dot where there is a red x.
    for step_i, (xv, ys) in enumerate(zip(x_positions, rewards)):
        killed = killed_by_step.get(step_i, set())
        alive_idx = [i for i in range(len(ys)) if i not in killed]
        yvj = yj_by_step[step_i]
        if alive_idx:
            ax.scatter(
                [xv for _ in alive_idx],
                [yvj[i] for i in alive_idx],
                s=20,
                color="#1f77b4",
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
            )
        if killed and timesteps[step_i] != 0:
            ax.scatter(
                [xv for _ in sorted(killed)],
                [yvj[i] for i in sorted(killed)],
                marker="x",
                s=70,
                color="red",
                linewidths=2,
                zorder=6,
            )

    # Parent->child arrows and kills at each resampling step.
    for step_idx in range(1, len(timesteps)):
        prev_y = rewards[step_idx - 1]
        cur_y = rewards[step_idx]
        prev_full_idx = selected_indices[step_idx - 1]
        cur_parents = (
            parents_all[prev_full_idx]
            if prev_full_idx < len(parents_all)
            else list(range(n_particles))
        )
        x_prev = x_positions[step_idx - 1]
        x_cur = x_positions[step_idx]

        for child_idx in range(min(n_particles, len(cur_y))):
            parent_idx = cur_parents[child_idx] if child_idx < len(cur_parents) else child_idx
            if parent_idx < 0 or parent_idx >= len(prev_y):
                parent_idx = child_idx
            ax.annotate(
                "",
                xy=(x_cur, yj_by_step[step_idx][child_idx]),
                xytext=(x_prev, yj_by_step[step_idx - 1][parent_idx]),
                arrowprops=dict(arrowstyle="->", color="#666666", lw=1.4, shrinkA=0, shrinkB=0),
                zorder=2,
            )

    y_final = rewards[-1]
    best_idx = int(np.argmax(y_final))
    y_best = yj_by_step[-1][best_idx]
    ax.scatter(
        [x_positions[-1]],
        [y_best],
        s=30,
        color="#2ca02c",
        edgecolors="black",
        linewidths=0.9,
        zorder=8,
    )

    # Final thumbnails (all survivors), ordered by final reward.
    sample_paths = [Path(p) for p in fk_traj.get("sample_paths", [])]
    if y_final and sample_paths:
        y_low = min(y_bounds[0], min(y_final)) if y_bounds is not None else min(y_final)
        y_high = max(y_bounds[1], max(y_final)) if y_bounds is not None else max(y_final)
        thumb_gap = _stack_gap_from_bounds(y_bounds, (y_low, y_high))
        ranking = sorted(range(len(y_final)), key=lambda i: y_final[i], reverse=True)
        top_y = y_final[ranking[0]]
        stack_ys = _stack_thumbnail_y_positions(
            top_y=top_y,
            n_items=len(ranking),
            gap=thumb_gap,
            y_bounds=y_bounds,
            y_fallback=(y_low, y_high),
        )
        for rank, idx in enumerate(ranking):
            if idx >= len(sample_paths) or not sample_paths[idx].exists():
                continue
            y_thumb = stack_ys[rank]
            _add_thumb(
                ax,
                _load_thumb_array(sample_paths[idx]),
                100.0 + thumb_x_offset,
                y_thumb,
                thumb_zoom,
                edgecolor="#2ca02c" if rank == 0 else "black",
                linewidth=1.6 if rank == 0 else 1.0,
            )

    if y_bounds is None:
        y_min, y_max = min(y_final), max(y_final)
        y_pad = 0.08 * max(1e-6, y_max - y_min)
        y_bounds = (y_min - y_pad, y_max + y_pad)
    x_min = min(0.0, min(x_positions)) - 4.0
    ax.set_xlim(x_min, 100.0 + thumb_x_offset + 1.0)
    ax.set_ylim(y_bounds[0], y_bounds[1])


def _plot_pps_panel(
    ax: plt.Axes,
    prompt_df: pd.DataFrame,
    sample_map: dict[int, Path],
    *,
    k_init: int,
    cutoff_times: list[int],
    remaining_particles: list[int],
    total_steps: int,
    thumb_zoom: float,
    thumb_x_offset: float,
    y_bounds: tuple[float, float] | None = None,
    jitter_map: dict[str, dict[int, float]] | None = None,
    repeat_xticks: bool = False,
) -> None:
    jitter_map = jitter_map or {}
    if len(cutoff_times) != len(remaining_particles):
        raise ValueError("--cutoff-times and --remaining-particles must have same length.")
    stage_steps = [1] + cutoff_times + [total_steps]
    x = _progress_positions(stage_steps, total_steps)
    xlabels = _progress_labels(stage_steps, total_steps)
    _style_panel(
        ax,
        "Progressive Seed Pruning",
        x,
        xlabels,
        y_label=False,
        show_xlabel=False,
        show_xticklabels=repeat_xticks,
    )
    x_by_step = {st: xp for st, xp in zip(stage_steps, x)}

    pool = prompt_df[
        (prompt_df["local_seed_idx"] >= 0) & (prompt_df["local_seed_idx"] < k_init)
    ].copy()
    if pool.empty:
        raise ValueError("No PPS candidates found for prompt.")

    seed_ids = sorted(pool["seed"].astype(int).unique().tolist())
    survivors = seed_ids[:]
    killed_at_stage: dict[int, set[int]] = {}
    survivor_sets: dict[int, set[int]] = {1: set(survivors)}

    for cutoff, keep in zip(cutoff_times, remaining_particles):
        at_cutoff = pool[(pool["step"] == cutoff) & (pool["seed"].isin(survivors))]
        at_cutoff = at_cutoff.sort_values(["image_reward", "seed"], ascending=[False, True])
        kept = at_cutoff["seed"].head(keep).astype(int).tolist()
        killed_at_stage[cutoff] = set(survivors) - set(kept)
        survivors = kept
        survivor_sets[cutoff] = set(survivors)
    survivor_sets[total_steps] = set(survivors)

    final_rows = pool[(pool["step"] == total_steps) & (pool["seed"].isin(survivors))].sort_values(
        ["image_reward", "seed"], ascending=[False, True]
    )
    if final_rows.empty:
        raise ValueError("No final survivors for PPS.")
    winner_seed = int(final_rows.iloc[0]["seed"])

    y_min = float(pool["image_reward"].min())
    y_max = float(pool["image_reward"].max())
    y_pad = 0.08 * max(1e-6, y_max - y_min)

    for seed in seed_ids:
        seed_rank = seed_ids.index(seed)
        one = pool[pool["seed"] == seed]
        xs: list[float] = []
        ys: list[float] = []
        ysj: list[float] = []
        alive = True
        for st in stage_steps:
            if not alive:
                break
            row = one[one["step"] == st]
            if row.empty:
                break
            yv = float(row.iloc[0]["image_reward"])
            xs.append(float(x_by_step[st]))
            ysj.append(
                _dot_y_with_jitter(
                    x=float(x_by_step[st]),
                    y=yv,
                    algorithm="pps",
                    seed_idx=seed_rank,
                    n_seeds=len(seed_ids),
                    jitter_map=jitter_map,
                )
            )
            ys.append(yv)
            if st in cutoff_times and seed in killed_at_stage.get(st, set()):
                # Killed point: only red x.
                ax.scatter([xs[-1]], [ysj[-1]], marker="x", s=70, color="red", linewidths=2, zorder=6)
                alive = False
        if xs:
            _draw_arrow_path(ax, xs, ysj, color="#666666", lw=1.4)
            # Black dots for all alive intermediate points.
            if len(xs) > 1:
                # First point always shown as a dot.
                ax.scatter([xs[0]], [ysj[0]], s=20, color="#1f77b4", edgecolors="black", linewidths=0.8, zorder=4)
                if len(xs) > 2:
                    ax.scatter(xs[1:-1], ysj[1:-1], s=20, color="#1f77b4", edgecolors="black", linewidths=0.8, zorder=4)
            if alive:
                ax.scatter([xs[-1]], [ysj[-1]], s=24, color="#1f77b4", edgecolors="black", linewidths=0.8, zorder=7)

    winner_row = final_rows.iloc[0]
    # Ensure winner endpoint is a black dot (not a star).
    ax.scatter(
        [x[-1]],
        [float(winner_row["image_reward"])],
        s=30,
        color="#2ca02c",
        edgecolors="black",
        linewidths=0.9,
        zorder=8,
    )
    # Show thumbnails for all final survivors (ordered by final reward).
    thumb_gap = _stack_gap_from_bounds(y_bounds, (y_min, y_max))
    top_final_y = float(final_rows.iloc[0]["image_reward"])
    stack_ys = _stack_thumbnail_y_positions(
        top_y=top_final_y,
        n_items=len(final_rows),
        gap=thumb_gap,
        y_bounds=y_bounds,
        y_fallback=(y_min, y_max),
    )
    for rank, row in enumerate(final_rows.itertuples(index=False)):
        seed = int(row.seed)
        img_path = sample_map.get(seed)
        if img_path is None:
            continue
        y_thumb = stack_ys[rank]
        _add_thumb(
            ax,
            _load_thumb_array(img_path),
            x[-1] + thumb_x_offset,
            y_thumb,
            thumb_zoom,
            edgecolor="#2ca02c" if rank == 0 else "black",
            linewidth=1.6 if rank == 0 else 1.0,
        )

    ax.set_xlim(min(x) - 2.0, x[-1] + thumb_x_offset + 1.0)
    if y_bounds is None:
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
    else:
        ax.set_ylim(y_bounds[0], y_bounds[1])


def _plot_bon_compute_panel(
    ax: plt.Axes,
    *,
    x_ticks: list[float],
    x_labels: list[str],
    n_particles: int,
    n_slots: int,
) -> None:
    _style_compute_panel(ax, x_ticks, x_labels)
    bar_h = 0.7
    n_slots = max(n_slots, n_particles)
    spacing = 1.7
    group_span = (n_particles - 1) * spacing + 1.0
    start = (n_slots - group_span) / 2.0
    y = start + np.arange(n_particles) * spacing
    ax.barh(y, [100.0] * n_particles, left=0.0, height=bar_h, color="#1f77b4", edgecolor="black")
    ax.set_ylim(-0.75, n_slots - 0.25)


def _plot_fk_compute_panel(
    ax: plt.Axes,
    fk_traj: dict[str, Any],
    *,
    n_particles: int,
    n_slots: int,
    continue_first_child: bool,
) -> None:
    timesteps_all = [int(v) for v in fk_traj.get("timesteps", [])]
    parents_all = [[int(x) for x in row] for row in fk_traj.get("parents", [])]
    total_steps = int(fk_traj.get("time_steps", 64))
    if not timesteps_all:
        _style_compute_panel(ax, [0.0, 100.0], ["0", "100"])
        ax.set_ylim(-0.75, max(n_slots, n_particles) - 0.25)
        return
    if len(parents_all) != len(timesteps_all):
        parents_all = [[i for i in range(n_particles)] for _ in timesteps_all]

    raw_resampling_steps = [int(v) for v in fk_traj.get("resampling_steps", [])]
    if not raw_resampling_steps:
        raw_resampling_steps = [int(v) for v in timesteps_all]
    selected_steps = sorted(set([0] + raw_resampling_steps + [max(timesteps_all)]))
    idx_by_step = {t: i for i, t in enumerate(timesteps_all)}
    selected_indices = [idx_by_step[s] for s in selected_steps if s in idx_by_step]
    if not selected_indices:
        selected_indices = [len(timesteps_all) - 1]
    timesteps = [timesteps_all[i] for i in selected_indices]
    x_positions = []
    for st in timesteps:
        if total_steps <= 1:
            pct = 100.0
        else:
            pct = 100.0 * float(st) / float(max(1, total_steps - 1))
        x_positions.append(float(_round_to_nearest_ten(pct)))
    # Deduplicate rounded checkpoints while preserving order.
    x_idx_pairs: list[tuple[float, int]] = []
    seen_x: set[float] = set()
    for x, idx in zip(x_positions, selected_indices):
        if x not in seen_x:
            x_idx_pairs.append((x, idx))
            seen_x.add(x)
    x_positions = [p[0] for p in x_idx_pairs]
    selected_indices = [p[1] for p in x_idx_pairs]

    tick_positions = sorted(set(x_positions))
    tick_labels = [str(int(v)) for v in tick_positions]
    _style_compute_panel(ax, tick_positions, tick_labels)

    n_slots = max(n_slots, n_particles)
    spacing = 1.7
    group_span = (n_particles - 1) * spacing + 1.0
    start = (n_slots - group_span) / 2.0
    y = start + np.arange(n_particles) * spacing
    bar_h = 0.7

    # Two modes:
    # - continue_first_child=True: lineage continuation via first child.
    # - continue_first_child=False: bars break at every checkpoint; all children reconnect via gray lines.
    n_tracks = n_particles
    x0 = x_positions[0]
    x_last = x_positions[-1]
    # Visible horizontal gap at kill/rebirth boundaries.
    kill_gap = 4.0
    # Collected segments per track.
    segments: list[list[tuple[float, float]]] = [[] for _ in range(n_tracks)]

    if continue_first_child:
        # Current track -> particle index at current checkpoint.
        track_particle: list[int] = list(range(n_tracks))
        # Current particle index -> track.
        particle_track: dict[int, int] = {p: p for p in range(n_tracks)}
        # Open segment start per track.
        seg_start: list[float | None] = [x0 for _ in range(n_tracks)]

        for seg_idx in range(len(x_positions) - 1):
            x_next = x_positions[seg_idx + 1]
            x_boundary = x_positions[seg_idx]
            prev_full_idx = selected_indices[seg_idx]
            parent_map = (
                parents_all[prev_full_idx]
                if prev_full_idx < len(parents_all)
                else list(range(n_particles))
            )

            # Children grouped by current track (via parent particle -> parent track).
            children_by_track: dict[int, list[int]] = {t: [] for t in range(n_tracks)}
            for child_idx in range(n_particles):
                parent_particle = parent_map[child_idx] if child_idx < len(parent_map) else child_idx
                parent_track = particle_track.get(parent_particle, parent_particle)
                parent_track = min(max(parent_track, 0), n_tracks - 1)
                children_by_track[parent_track].append(child_idx)

            # Determine continuation and extra births.
            next_track_particle: list[int | None] = [None for _ in range(n_tracks)]
            dead_tracks: list[int] = []
            extra_children: list[int] = []
            for t in range(n_tracks):
                kids = children_by_track.get(t, [])
                if kids:
                    # First child continues this track.
                    next_track_particle[t] = kids[0]
                    # Additional children are born on tracks that became available.
                    extra_children.extend(kids[1:])
                else:
                    dead_tracks.append(t)

            # Close killed tracks at this checkpoint boundary.
            for t in dead_tracks:
                if seg_start[t] is not None:
                    end_x = max(seg_start[t], x_boundary - kill_gap / 2.0)
                    segments[t].append((seg_start[t], end_x))
                seg_start[t] = None

            # Rebirth extra children on dead tracks at this checkpoint boundary.
            n_birth = min(len(dead_tracks), len(extra_children))
            for i in range(n_birth):
                t = dead_tracks[i]
                c = extra_children[i]
                next_track_particle[t] = c
                seg_start[t] = min(x_last, x_boundary + kill_gap / 2.0)

            # Any continuing track keeps its segment open; any newly used track starts at boundary.
            for t in range(n_tracks):
                if next_track_particle[t] is not None and seg_start[t] is None:
                    seg_start[t] = x_boundary

            # Draw gray parent->child connectors at this boundary.
            child_track: dict[int, int] = {}
            for t, p in enumerate(next_track_particle):
                if p is not None:
                    child_track[int(p)] = t
            for child_idx in range(n_particles):
                parent_particle = parent_map[child_idx] if child_idx < len(parent_map) else child_idx
                parent_t = particle_track.get(parent_particle, parent_particle)
                child_t = child_track.get(child_idx)
                if child_t is None:
                    continue
                parent_t = min(max(int(parent_t), 0), n_tracks - 1)
                child_t = min(max(int(child_t), 0), n_tracks - 1)
                if parent_t == child_t:
                    continue
                x_parent = x_boundary - kill_gap / 2.0
                x_child = x_boundary + kill_gap / 2.0
                ax.plot(
                    [x_parent, x_child],
                    [y[parent_t], y[child_t]],
                    color="#666666",
                    linewidth=1.2,
                    zorder=1,
                )

            # Build reverse mapping for next checkpoint.
            particle_track = {}
            for t, p in enumerate(next_track_particle):
                if p is not None:
                    particle_track[int(p)] = t
                    track_particle[t] = int(p)

        # Close remaining live segments at final checkpoint.
        for t in range(n_tracks):
            if seg_start[t] is not None:
                segments[t].append((seg_start[t], x_last))
    else:
        # Break-at-every-checkpoint mode.
        # Tracks correspond to particle indices at each segment (no continuation).
        prev_particle_track: dict[int, int] = {i: i for i in range(n_particles)}
        for seg_idx in range(len(x_positions) - 1):
            x0_seg = x_positions[seg_idx]
            x1_seg = x_positions[seg_idx + 1]
            left_pad = 0.0 if seg_idx == 0 else kill_gap / 2.0
            right_pad = 0.0 if seg_idx == (len(x_positions) - 2) else kill_gap / 2.0
            start_x = x0_seg + left_pad
            end_x = max(start_x, x1_seg - right_pad)
            for t in range(n_tracks):
                segments[t].append((start_x, end_x))

            # Connect every child (including the first child) to parent at boundary.
            prev_full_idx = selected_indices[seg_idx]
            parent_map = (
                parents_all[prev_full_idx]
                if prev_full_idx < len(parents_all)
                else list(range(n_particles))
            )
            # Assign children to tracks in parent-order to reduce line crossings.
            child_parent_tracks: list[tuple[int, int]] = []
            for child_idx in range(n_particles):
                parent_idx = parent_map[child_idx] if child_idx < len(parent_map) else child_idx
                parent_idx = min(max(int(parent_idx), 0), n_particles - 1)
                parent_track = prev_particle_track.get(parent_idx, parent_idx)
                child_parent_tracks.append((int(parent_track), child_idx))
            child_parent_tracks.sort(key=lambda t: (t[0], t[1]))
            child_track_map: dict[int, int] = {}
            for new_track, (_, child_idx) in enumerate(child_parent_tracks):
                child_track_map[child_idx] = new_track

            x_parent = x0_seg - kill_gap / 2.0
            x_child = x0_seg + kill_gap / 2.0
            for child_idx in range(n_particles):
                parent_idx = parent_map[child_idx] if child_idx < len(parent_map) else child_idx
                parent_idx = min(max(int(parent_idx), 0), n_tracks - 1)
                parent_track = prev_particle_track.get(parent_idx, parent_idx)
                parent_track = min(max(int(parent_track), 0), n_tracks - 1)
                child_track = child_track_map.get(child_idx, child_idx)
                child_track = min(max(int(child_track), 0), n_tracks - 1)
                ax.plot(
                    [x_parent, x_child],
                    [y[parent_track], y[child_track]],
                    color="#666666",
                    linewidth=1.2,
                    zorder=1,
                )
            prev_particle_track = {child_idx: child_track_map[child_idx] for child_idx in range(n_particles)}

    # Draw lineage segments.
    for t in range(n_tracks):
        for start_x, end_x in segments[t]:
            width = max(0.0, end_x - start_x)
            if width <= 0:
                continue
            ax.barh(
                [y[t]],
                [width],
                left=start_x,
                height=bar_h,
                color="#1f77b4",
                edgecolor="black",
                zorder=2,
            )

    ax.set_ylim(-0.75, n_slots - 0.25)


def _plot_pps_compute_panel(
    ax: plt.Axes,
    prompt_df: pd.DataFrame,
    *,
    x_ticks: list[float],
    x_labels: list[str],
    k_init: int,
    cutoff_times: list[int],
    remaining_particles: list[int],
    total_steps: int,
) -> None:
    _style_compute_panel(ax, x_ticks, x_labels)
    if len(cutoff_times) != len(remaining_particles):
        raise ValueError("--cutoff-times and --remaining-particles must have same length.")
    pool = prompt_df[
        (prompt_df["local_seed_idx"] >= 0) & (prompt_df["local_seed_idx"] < k_init)
    ].copy()
    seed_ids = sorted(pool["seed"].astype(int).unique().tolist())
    if not seed_ids:
        raise ValueError("No PPS candidates found for compute panel.")
    survivors = seed_ids[:]
    end_step_by_seed: dict[int, int] = {s: total_steps for s in seed_ids}
    for cutoff, keep in zip(cutoff_times, remaining_particles):
        at_cutoff = pool[(pool["step"] == cutoff) & (pool["seed"].isin(survivors))]
        at_cutoff = at_cutoff.sort_values(["image_reward", "seed"], ascending=[False, True])
        kept = at_cutoff["seed"].head(keep).astype(int).tolist()
        killed = set(survivors) - set(kept)
        for s in killed:
            end_step_by_seed[s] = cutoff
        survivors = kept
    widths = [100.0 * float(end_step_by_seed[s]) / float(max(1, total_steps)) for s in seed_ids]
    # Sort compute bars by duration: longest at top, shortest at bottom.
    widths = sorted(widths, reverse=True)
    bar_h = 0.7
    y = np.arange(len(seed_ids))[::-1]
    ax.barh(y, widths, left=0.0, height=bar_h, color="#1f77b4", edgecolor="black")
    ax.set_ylim(-0.75, len(seed_ids) - 0.25)


def main() -> None:
    args = parse_args()
    jitter_map = parse_dot_jitter(args.dot_jitter)
    rewards_root, fk_root, total_steps, cutoff_times = _resolve_paths(args)
    remaining_particles = parse_int_list(args.remaining_particles)
    if len(cutoff_times) != len(remaining_particles):
        raise ValueError("--cutoff-times and --remaining-particles must have same length.")

    metadata_path = Path(args.metadata_path).resolve()
    prompt_text = _read_metadata_prompt(metadata_path, args.prompt_id)

    prompt_df = _load_prompt_traj(rewards_root, args.prompt_id)
    sample_map = _sample_path_map_for_prompt(rewards_root, args.prompt_id)
    fk_seed_dir = _resolve_fk_seed_dir(fk_root, args.fk_seed_dir, args.fk_seed_id)
    fk_prompt_dir = fk_seed_dir / f"{args.prompt_id:05d}"
    if not fk_prompt_dir.exists():
        raise FileNotFoundError(f"FK prompt directory not found: {fk_prompt_dir}")
    if args.no_fk_intermediates:
        fk_traj = _load_fk_final_only(fk_prompt_dir=fk_prompt_dir, fk_seed_dir=fk_seed_dir)
    else:
        repo_root = Path(__file__).resolve().parents[2]
        fk_traj = _collect_fk_trajectory_live(
            repo_root=repo_root,
            fk_seed_dir=fk_seed_dir,
            prompt_id=args.prompt_id,
            prompt_text=prompt_text,
            seed_id=args.fk_seed_id,
            force_rerun=args.force_fk_rerun,
        )
    fk_vals = [float(v) for step in fk_traj.get("rewards", []) for v in step]
    if not fk_vals:
        raise ValueError("FK trajectory has no rewards to plot.")

    bon_pool = prompt_df[(prompt_df["local_seed_idx"] >= 0) & (prompt_df["local_seed_idx"] < args.k_bon)]
    pps_pool = prompt_df[(prompt_df["local_seed_idx"] >= 0) & (prompt_df["local_seed_idx"] < args.k_init)]
    y_all: list[float] = []
    if not bon_pool.empty:
        y_all.extend([float(v) for v in bon_pool["image_reward"].tolist()])
    if not pps_pool.empty:
        y_all.extend([float(v) for v in pps_pool["image_reward"].tolist()])
    y_all.extend([float(v) for v in fk_vals])
    if not y_all:
        raise ValueError("No reward values available to determine shared y-axis bounds.")
    y_min = min(y_all)
    y_max = max(y_all)
    y_pad = 0.08 * max(1e-6, y_max - y_min)
    shared_y_bounds = (y_min - y_pad, y_max + y_pad)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(9.2, 5.2),
        sharex="col",
        gridspec_kw={"height_ratios": [3.0, 1.4]},
    )
    plt.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.16, wspace=0.48, hspace=0.18)

    _plot_bon_panel(
        axes[0, 0],
        prompt_df,
        sample_map,
        k_bon=args.k_bon,
        total_steps=total_steps,
        cutoff_times=cutoff_times,
        thumb_zoom=args.thumb_zoom,
        thumb_x_offset=args.thumb_x_offset,
        y_bounds=shared_y_bounds,
        jitter_map=jitter_map,
        repeat_xticks=args.repeat_xticks,
    )
    bon_stage_steps = [1] + cutoff_times + [total_steps]
    bon_extra_step = int(round(0.75 * total_steps))
    if 1 < bon_extra_step < total_steps:
        bon_stage_steps.append(bon_extra_step)
    bon_stage_steps = sorted(set(bon_stage_steps))
    bon_x = _progress_positions(bon_stage_steps, total_steps)
    bon_labels = _progress_labels(bon_stage_steps, total_steps)
    _plot_bon_compute_panel(
        axes[1, 0],
        x_ticks=bon_x,
        x_labels=bon_labels,
        n_particles=args.k_bon,
        n_slots=args.k_init,
    )

    _plot_fk_panel(
        axes[0, 1],
        fk_traj,
        thumb_zoom=args.thumb_zoom,
        thumb_x_offset=args.thumb_x_offset,
        y_bounds=shared_y_bounds,
        jitter_map=jitter_map,
        repeat_xticks=args.repeat_xticks,
    )
    _plot_fk_compute_panel(
        axes[1, 1],
        fk_traj,
        n_particles=args.k_bon,
        n_slots=args.k_init,
        continue_first_child=args.fk_bars_continue_first_child,
    )
    axes[1, 0].set_ylabel("Compute Allocated\nper Trajectory\n", fontsize=11)

    _plot_pps_panel(
        axes[0, 2],
        prompt_df,
        sample_map,
        k_init=args.k_init,
        cutoff_times=cutoff_times,
        remaining_particles=remaining_particles,
        total_steps=total_steps,
        thumb_zoom=args.thumb_zoom,
        thumb_x_offset=args.thumb_x_offset,
        y_bounds=shared_y_bounds,
        jitter_map=jitter_map,
        repeat_xticks=args.repeat_xticks,
    )
    pps_stage_steps = [1] + cutoff_times + [total_steps]
    pps_x = _progress_positions(pps_stage_steps, total_steps)
    pps_labels = _progress_labels(pps_stage_steps, total_steps)
    _plot_pps_compute_panel(
        axes[1, 2],
        prompt_df,
        x_ticks=pps_x,
        x_labels=pps_labels,
        k_init=args.k_init,
        cutoff_times=cutoff_times,
        remaining_particles=remaining_particles,
        total_steps=total_steps,
    )

    fig.text(
        0.5,
        0.94,
        f'Prompt: "{prompt_text}"',
        fontsize=13,
        ha="center",
        va="top",
    )

    default_stem = f"methods_trajectories_{args.model}_prompt_{args.prompt_id:04d}"
    out_png = Path(args.out_png or f"paper_figures/figures/{default_stem}.png").resolve()
    out_pdf = Path(args.out_pdf or f"paper_figures/figures/{default_stem}.pdf").resolve()
    out_png_generic = out_png.parent / "methods_trajectories.png"
    out_pdf_generic = out_pdf.parent / "methods_trajectories.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_png_generic, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_pdf_generic, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"FK seed dir used: {fk_seed_dir}")
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved PNG: {out_png_generic}")
    print(f"Saved PDF: {out_pdf_generic}")


if __name__ == "__main__":
    main()
