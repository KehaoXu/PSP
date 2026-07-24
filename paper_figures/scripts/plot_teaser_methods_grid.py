#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


MODEL_CONFIG: dict[str, dict[str, Any]] = {
    "sd15": {
        "name": "Stable Diffusion v1.5",
        "rewards_root": "results/reward_signal/sd15_reward_signal",
        "fk_root": "results/fk_steering/sd15/fk_sdv15_k4_t64_eta1_ir_geneval",
        "geneval_csv": "results/reward_signal/sd15_reward_signal/sd15_geneval_sample_scores.csv",
        "total_steps": 64,
        "cutoff_times": [16, 32],
    },
    "sdxl": {
        "name": "Stable Diffusion XL",
        "rewards_root": "results/reward_signal/sdxl_reward_signal",
        "fk_root": "results/fk_steering/sdxl/fk_sdxl_k4_t64_eta1_ir_geneval",
        "geneval_csv": "results/reward_signal/sdxl_reward_signal/sdxl_geneval_sample_scores.csv",
        "total_steps": 64,
        "cutoff_times": [16, 32],
    },
    "sd35": {
        "name": "Stable Diffusion 3.5",
        "rewards_root": "results/reward_signal/sd35_reward_signal",
        "fk_root": "results/fk_steering/sd35/fk_sdv35_k4_t32_gamma_0p005_ir_geneval",
        "geneval_csv": "results/reward_signal/sd35_reward_signal/sd35_geneval_sample_scores.csv",
        "total_steps": 32,
        "cutoff_times": [8, 16],
    },
}


def parse_int_list(value: str) -> list[int]:
    vals = [x.strip() for x in value.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return [int(v) for v in vals]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 4xN teaser grid for regular/BoN/FK/PSP rows."
    )
    parser.add_argument(
        "--prompt-ids",
        type=str,
        default="92,230,436,394,500",
        help='Comma-separated prompt ids (expected 5), e.g. "0,12,42,128,300".',
    )
    parser.add_argument(
        "--random-prompt-seed",
        type=int,
        default=None,
        help=(
            "If set, ignore --prompt-ids and sample 5 prompt ids uniformly "
            "without replacement from metadata using this seed."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=sorted(MODEL_CONFIG.keys()),
        default="sdxl",
    )
    parser.add_argument(
        "--rewards-root",
        type=str,
        default="",
        help="Override default rewards root for selected model.",
    )
    parser.add_argument(
        "--fk-root",
        type=str,
        default="",
        help="Override default FK run root for selected model.",
    )
    parser.add_argument(
        "--geneval-csv",
        type=str,
        default="",
        help="Override default Geneval sample score CSV for selected model.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl",
    )
    parser.add_argument(
        "--fk-seed-dir",
        type=str,
        default="",
        help="Optional FK seed directory name (e.g., seed=42_20260213-084446).",
    )
    parser.add_argument(
        "--k-bon",
        type=int,
        default=4,
        help="BoN pool size (default: 4).",
    )
    parser.add_argument(
        "--k-init",
        type=int,
        default=8,
        help="Initial K for PSP scheduled pruning.",
    )
    parser.add_argument(
        "--label-col-ratio",
        type=float,
        default=1.4,
        help="Width ratio of the first (row-label) column.",
    )
    parser.add_argument(
        "--remaining-particles",
        type=str,
        default="4,2",
        help='PSP survivors after each cutoff, e.g. "4,2".',
    )
    parser.add_argument(
        "--exclude-rows",
        type=str,
        default="",
        help='Comma-separated rows to exclude: "inference,bon,fk,psp" (or legacy "pps").',
    )
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Shortcut to exclude the regular inference row.",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default="paper_figures/figures/teaser_methods_grid.png",
    )
    parser.add_argument(
        "--out-pdf",
        type=str,
        default="paper_figures/figures/teaser_methods_grid.pdf",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Legacy DPI used if specific values are not provided.")
    parser.add_argument(
        "--png-dpi",
        type=int,
        default=450,
        help="DPI for PNG export.",
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=600,
        help="Raster DPI for images embedded in PDF export.",
    )
    parser.add_argument(
        "--quality-scale",
        type=float,
        default=1.0,
        help=(
            "Approximate output size scaling factor. "
            "Example: 0.25 targets ~25%% file size by reducing effective DPI."
        ),
    )
    parser.add_argument(
        "--separator-x-start",
        type=float,
        default=0.12,
        help="Separator line start x in figure coords [0,1].",
    )
    parser.add_argument(
        "--separator-x-end-pad",
        type=float,
        default=0.003,
        help="Extra right padding added to last image-column x end.",
    )
    parser.add_argument(
        "--separator-linewidth",
        type=float,
        default=4.0,
        help="Separator line thickness.",
    )
    return parser.parse_args()


def _defaulted_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    cfg = MODEL_CONFIG[args.model]
    rewards_root = Path(args.rewards_root or cfg["rewards_root"]).resolve()
    fk_root = Path(args.fk_root or cfg["fk_root"]).resolve()
    geneval_csv = Path(args.geneval_csv or cfg["geneval_csv"]).resolve()
    return rewards_root, fk_root, geneval_csv


def _read_metadata_rows(metadata_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No metadata rows found in {metadata_path}")
    return rows


def _load_final_metrics(rewards_root: Path, total_steps: int) -> pd.DataFrame:
    metric_paths = sorted(rewards_root.glob("*geneval*/metrics.csv"))
    if not metric_paths:
        raise FileNotFoundError(f"No geneval metrics.csv found under {rewards_root}")

    parts = [pd.read_csv(p, usecols=["prompt_id", "seed", "step", "image_reward"]) for p in metric_paths]
    df = pd.concat(parts, ignore_index=True)
    final_df = df[df["step"] == total_steps].copy()
    if final_df.empty:
        raise ValueError(f"No rows found at final step={total_steps} under {rewards_root}")
    final_df = final_df.drop_duplicates(subset=["prompt_id", "seed"], keep="last")
    pair_idx = (
        final_df[["prompt_id", "seed"]]
        .drop_duplicates()
        .sort_values(["prompt_id", "seed"])
        .reset_index(drop=True)
    )
    pair_idx["local_seed_idx"] = pair_idx.groupby("prompt_id").cumcount()
    return final_df.merge(pair_idx, on=["prompt_id", "seed"], how="inner")


def _build_sample_path_map(rewards_root: Path) -> dict[tuple[int, int], Path]:
    mapping: dict[tuple[int, int], Path] = {}
    sample_roots = sorted(rewards_root.glob("*geneval*/samples"))
    for sample_root in sample_roots:
        for prompt_dir in sorted(sample_root.iterdir()):
            if not prompt_dir.is_dir() or not prompt_dir.name.isdigit():
                continue
            prompt_id = int(prompt_dir.name)
            for image_path in sorted(prompt_dir.glob("*.png")):
                if not image_path.stem.isdigit():
                    continue
                seed = int(image_path.stem)
                mapping[(prompt_id, seed)] = image_path
    if not mapping:
        raise ValueError(f"No sample images found under {rewards_root}")
    return mapping


def _scheduled_pick_seed(
    prompt_df: pd.DataFrame,
    *,
    k_init: int,
    logical_seed: int,
    cutoff_times: list[int],
    remaining_particles: list[int],
    total_steps: int,
) -> int:
    start = logical_seed * k_init
    end = (logical_seed + 1) * k_init - 1
    pool = prompt_df[
        (prompt_df["local_seed_idx"] >= start) & (prompt_df["local_seed_idx"] <= end)
    ].copy()
    if pool.empty:
        raise ValueError(
            f"No seeds for prompt_id={int(prompt_df['prompt_id'].iloc[0])} in window {start}..{end}."
        )

    survivors = sorted(pool["seed"].astype(int).unique().tolist())
    steps_df = prompt_df[prompt_df["seed"].isin(survivors)].copy()
    for cutoff, keep in zip(cutoff_times, remaining_particles):
        at_cutoff = steps_df[steps_df["step"] == cutoff].sort_values(
            ["image_reward", "seed"], ascending=[False, True]
        )
        if at_cutoff.empty:
            raise ValueError(f"Missing cutoff step={cutoff} for prompt_id={int(prompt_df['prompt_id'].iloc[0])}")
        survivors = at_cutoff["seed"].head(keep).astype(int).tolist()
        steps_df = prompt_df[prompt_df["seed"].isin(survivors)].copy()

    at_final = prompt_df[(prompt_df["step"] == total_steps) & (prompt_df["seed"].isin(survivors))].sort_values(
        ["image_reward", "seed"], ascending=[False, True]
    )
    if at_final.empty:
        raise ValueError(f"Missing final survivors for prompt_id={int(prompt_df['prompt_id'].iloc[0])}")
    return int(at_final.iloc[0]["seed"])


def _resolve_fk_seed_dir(fk_root: Path, requested: str) -> Path:
    if requested:
        path = fk_root / requested
        if not path.exists():
            raise FileNotFoundError(f"--fk-seed-dir not found: {path}")
        return path
    candidates = sorted(p for p in fk_root.glob("seed=*_*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No seed=*_* directories found in FK root: {fk_root}")
    return candidates[0]


def _fk_winner_path(seed_dir: Path, prompt_id: int) -> Path:
    prompt_dir = seed_dir / f"{prompt_id:05d}"
    winners = sorted((prompt_dir / "best_of_n_samples").glob("*.png"))
    if not winners:
        raise FileNotFoundError(f"No FK winner image found for prompt {prompt_id} in {prompt_dir}")
    return winners[0]


def _model_display_name(model_key: str) -> str:
    return str(MODEL_CONFIG[model_key]["name"])


def _wrap_prompt_title(text: str, width: int = 24) -> str:
    clean = " ".join(text.split())
    if not clean:
        return ""
    return "\n".join(textwrap.wrap(clean, width=width))


def _parse_excluded_rows(exclude_rows: str, *, no_inference: bool) -> set[str]:
    allowed = {"inference", "bon", "fk", "psp", "pps"}
    out: set[str] = set()
    if exclude_rows.strip():
        out = {x.strip().lower() for x in exclude_rows.split(",") if x.strip()}
        bad = sorted(out - allowed)
        if bad:
            raise ValueError(f"Unknown values in --exclude-rows: {bad}. Allowed: {sorted(allowed)}")
        # Normalize legacy spelling to new method name.
        if "pps" in out:
            out.remove("pps")
            out.add("psp")
    if no_inference:
        out.add("inference")
    return out


def main() -> None:
    args = parse_args()
    remaining_particles = parse_int_list(args.remaining_particles)
    if remaining_particles != [4, 2]:
        raise ValueError("This teaser is specified for ks=4,2 only. Use --remaining-particles 4,2.")

    cfg = MODEL_CONFIG[args.model]
    total_steps = int(cfg["total_steps"])
    cutoff_times = [int(x) for x in cfg["cutoff_times"]]

    rewards_root, fk_root, _ = _defaulted_paths(args)
    metadata_path = Path(args.metadata_path).resolve()
    if not rewards_root.exists():
        raise FileNotFoundError(f"rewards root not found: {rewards_root}")
    if not fk_root.exists():
        raise FileNotFoundError(f"FK root not found: {fk_root}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")

    metrics_final = _load_final_metrics(rewards_root, total_steps=total_steps)
    sample_map = _build_sample_path_map(rewards_root)
    step_metrics = pd.concat(
        [pd.read_csv(p, usecols=["prompt_id", "seed", "step", "image_reward"]) for p in sorted(rewards_root.glob("*geneval*/metrics.csv"))],
        ignore_index=True,
    ).drop_duplicates(subset=["prompt_id", "seed", "step"], keep="last")
    pair_idx = (
        step_metrics[["prompt_id", "seed"]]
        .drop_duplicates()
        .sort_values(["prompt_id", "seed"])
        .reset_index(drop=True)
    )
    pair_idx["local_seed_idx"] = pair_idx.groupby("prompt_id").cumcount()
    step_metrics = step_metrics.merge(pair_idx, on=["prompt_id", "seed"], how="inner")

    metadata_rows = _read_metadata_rows(metadata_path)
    fk_seed_dir = _resolve_fk_seed_dir(fk_root, args.fk_seed_dir)

    if args.random_prompt_seed is None:
        if not args.prompt_ids.strip():
            raise ValueError("Provide --prompt-ids or --random-prompt-seed.")
        prompt_ids = parse_int_list(args.prompt_ids)
        if len(prompt_ids) != 5:
            raise ValueError(f"Expected exactly 5 prompt ids, got {len(prompt_ids)}: {prompt_ids}")
    else:
        if args.prompt_ids.strip():
            print("Ignoring --prompt-ids because --random-prompt-seed was provided.")
        if len(metadata_rows) < 5:
            raise ValueError(
                f"Need at least 5 metadata prompts for random selection, found {len(metadata_rows)}."
            )
        rng = np.random.default_rng(int(args.random_prompt_seed))
        prompt_ids = [int(v) for v in rng.choice(len(metadata_rows), size=5, replace=False).tolist()]
        print(f"Randomly selected prompt ids (seed={args.random_prompt_seed}): {prompt_ids}")

    model_name = _model_display_name(args.model)
    row_keys = ["inference", "bon", "fk", "psp"]
    row_labels_all = [
        model_name,
        f"{model_name}\n+Best-of-N",
        f"{model_name}\n+FK-Steering",
        f"{model_name}\n+PSP\n(ours)",
    ]

    rows_images: list[list[Path]] = [[], [], [], []]
    col_titles: list[str] = []

    for pid in prompt_ids:
        if pid < 0 or pid >= len(metadata_rows):
            raise ValueError(f"prompt_id out of range for metadata: {pid}")
        prompt_text = str(metadata_rows[pid].get("prompt", ""))
        col_titles.append(_wrap_prompt_title(prompt_text, width=23))

        # Row 1: regular inference (logical seed 0 => first prompt-local seed).
        reg_pool = metrics_final[
            (metrics_final["prompt_id"] == pid) & (metrics_final["local_seed_idx"] == 0)
        ].copy()
        if reg_pool.empty:
            raise ValueError(f"No regular (local seed idx 0) candidate for prompt={pid}")
        reg_seed = int(reg_pool.sort_values(["seed"]).iloc[0]["seed"])
        reg_path = sample_map.get((pid, reg_seed))
        if reg_path is None:
            raise FileNotFoundError(f"Regular image not found for prompt={pid}, seed={reg_seed}")
        rows_images[0].append(reg_path)

        # Row 2: BoN k=4 by final IR.
        bon_df = metrics_final[
            (metrics_final["prompt_id"] == pid)
            & (metrics_final["local_seed_idx"] >= 0)
            & (metrics_final["local_seed_idx"] <= args.k_bon - 1)
        ].copy()
        if bon_df.empty:
            raise ValueError(f"No BoN candidates for prompt={pid} in local seed range 0..{args.k_bon - 1}")
        bon_row = bon_df.sort_values(["image_reward", "seed"], ascending=[False, True]).iloc[0]
        bon_seed = int(bon_row["seed"])
        bon_path = sample_map.get((pid, bon_seed))
        if bon_path is None:
            raise FileNotFoundError(f"BoN image not found for prompt={pid}, seed={bon_seed}")
        rows_images[1].append(bon_path)

        # Row 3: FK winner image.
        rows_images[2].append(_fk_winner_path(fk_seed_dir, pid))

        # Row 4: PSP scheduled IR with k=8, ts=model default, ks=4,2.
        prompt_steps = step_metrics[step_metrics["prompt_id"] == pid].copy()
        psp_seed = _scheduled_pick_seed(
            prompt_steps,
            k_init=args.k_init,
            logical_seed=0,
            cutoff_times=cutoff_times,
            remaining_particles=remaining_particles,
            total_steps=total_steps,
        )
        psp_path = sample_map.get((pid, psp_seed))
        if psp_path is None:
            raise FileNotFoundError(f"PSP image not found for prompt={pid}, seed={psp_seed}")
        rows_images[3].append(psp_path)

    excluded_rows = _parse_excluded_rows(args.exclude_rows, no_inference=args.no_inference)
    selected_indices = [i for i, key in enumerate(row_keys) if key not in excluded_rows]
    if not selected_indices:
        raise ValueError("All rows were excluded. Keep at least one row to plot.")

    n_cols = len(prompt_ids)
    # Compact layout with a dedicated label column and optional spacer before PSP.
    use_separator = ("fk" not in excluded_rows) and ("psp" not in excluded_rows)
    n_plot_rows = len(selected_indices)
    n_grid_rows = n_plot_rows + (1 if use_separator else 0)

    # Choose figure size so image cells are approximately square.
    label_col_ratio = float(args.label_col_ratio)
    if label_col_ratio <= 0:
        raise ValueError("--label-col-ratio must be > 0.")
    spacer_row_ratio = 0.12
    image_row_total = float(n_plot_rows) + (spacer_row_ratio if use_separator else 0.0)
    desired_cell = 1.9
    fig_w = desired_cell * (n_cols + label_col_ratio)
    fig_h = desired_cell * image_row_total
    fig = plt.figure(figsize=(fig_w, fig_h))
    row_height_ratios = [1.0] * n_plot_rows
    if use_separator:
        fk_pos = next(pos for pos, idx in enumerate(selected_indices) if row_keys[idx] == "fk")
        row_height_ratios.insert(fk_pos + 1, spacer_row_ratio)
    grid = fig.add_gridspec(
        nrows=n_grid_rows,
        ncols=n_cols + 1,
        height_ratios=row_height_ratios,
        width_ratios=[label_col_ratio] + [1.0] * n_cols,
        hspace=0.05,
        wspace=0.0,
    )
    row_to_grid: dict[int, int] = {}
    g_row = 0
    for pos, idx in enumerate(selected_indices):
        row_to_grid[idx] = g_row
        if use_separator and row_keys[idx] == "fk":
            g_row += 2
        else:
            g_row += 1
    axes: dict[tuple[int, int], plt.Axes] = {}

    top_original_idx = selected_indices[0]
    for r in selected_indices:
        gr = row_to_grid[r]
        for c in range(n_cols):
            ax = fig.add_subplot(grid[gr, c + 1])
            axes[(r, c)] = ax
            with Image.open(rows_images[r][c]) as im:
                ax.imshow(
                    np.asarray(im.convert("RGB")),
                    interpolation="none",
                    resample=False,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == top_original_idx:
                ax.set_title(col_titles[c], fontsize=9)

    label_axes: dict[int, plt.Axes] = {}
    for r in selected_indices:
        label = row_labels_all[r]
        gr = row_to_grid[r]
        lax = fig.add_subplot(grid[gr, 0])
        lax.set_axis_off()
        lax.text(
            0.5,
            0.5,
            label,
            fontsize=13,
            fontweight="bold" if r == 3 else "normal",
            ha="center",
            va="center",
            transform=lax.transAxes,
        )
        label_axes[r] = lax

    if use_separator:
        fk_pos = next(pos for pos, idx in enumerate(selected_indices) if row_keys[idx] == "fk")
        spacer_ax = fig.add_subplot(grid[fk_pos + 1, :])
        spacer_ax.set_axis_off()
        y_sep = (spacer_ax.get_position().y0 + spacer_ax.get_position().y1) / 2.0
        left_row = selected_indices[0]
        fig.add_artist(
            plt.Line2D(
                [
                    max(0.0, min(1.0, args.separator_x_start)),
                    min(1.0, axes[(left_row, n_cols - 1)].get_position().x1 + args.separator_x_end_pad),
                ],
                [y_sep, y_sep],
                transform=fig.transFigure,
                color="black",
                linewidth=args.separator_linewidth,
                zorder=1000,
            )
        )
    out_png = Path(args.out_png).resolve()
    out_pdf = Path(args.out_pdf).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if args.quality_scale <= 0:
        raise ValueError("--quality-scale must be > 0.")
    dpi_scale = math.sqrt(float(args.quality_scale))
    png_dpi = int(args.png_dpi or args.dpi)
    pdf_dpi = int(args.pdf_dpi or args.dpi)
    png_dpi = max(72, int(round(png_dpi * dpi_scale)))
    pdf_dpi = max(72, int(round(pdf_dpi * dpi_scale)))
    fig.savefig(out_png, dpi=png_dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=pdf_dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"FK seed dir used: {fk_seed_dir}")
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


if __name__ == "__main__":
    main()
