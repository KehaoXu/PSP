#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tuning comparison figure as PNG and PDF."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper_figures/figures",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default="tuning_comparison",
        help="Base output filename (without extension).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PNG export.",
    )
    parser.add_argument(
        "--profile-x-jitter-max",
        "--profile-x-jitter-std",
        dest="profile_x_jitter_max",
        type=float,
        default=0.020,
        help="Max absolute additive jitter on profile x (uniform in [-max, +max]).",
    )
    parser.add_argument(
        "--profile-y-jitter-max",
        "--profile-y-jitter-std",
        dest="profile_y_jitter_max",
        type=float,
        default=0.100,
        help="Max absolute additive jitter on profile y (uniform in [-max, +max]).",
    )
    return parser.parse_args()


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _iter_strategy_items(path: Path, max_items: int | None = None):
    decoder = json.JSONDecoder()
    chunk_size = 1 << 20
    marker = '"strategies"'
    buf = ""
    yielded = 0

    with path.open("r", encoding="utf-8") as f:
        while True:
            idx = buf.find(marker)
            if idx >= 0:
                break
            chunk = f.read(chunk_size)
            if not chunk:
                return
            buf += chunk
            if len(buf) > 4 * chunk_size:
                buf = buf[-4 * chunk_size :]

        buf = buf[idx:]
        while True:
            lb = buf.find("[")
            if lb >= 0:
                buf = buf[lb + 1 :]
                break
            chunk = f.read(chunk_size)
            if not chunk:
                return
            buf += chunk

        eof = False
        while max_items is None or yielded < max_items:
            buf = buf.lstrip()
            if not buf and not eof:
                chunk = f.read(chunk_size)
                if chunk:
                    buf += chunk
                    continue
                eof = True
            if not buf and eof:
                return
            if buf.startswith("]"):
                return
            if buf.startswith(","):
                buf = buf[1:]
                continue

            try:
                item, used = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                if eof:
                    return
                chunk = f.read(chunk_size)
                if not chunk:
                    eof = True
                else:
                    buf += chunk
                continue

            yield item
            yielded += 1
            buf = buf[used:]


def _profile_points(
    *,
    total_steps: int,
    k_init: int,
    cutoff_times: list[int],
    remaining_particles: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    cutoffs = [int(t) for t in cutoff_times]
    keeps = [int(k) for k in remaining_particles]
    if len(cutoffs) != len(keeps):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if total_steps <= 0 or k_init <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    x_steps = np.arange(0, int(total_steps) + 1, dtype=np.float64)
    y = np.full_like(x_steps, fill_value=float(k_init), dtype=np.float64)

    for t, keep in zip(cutoffs, keeps):
        t = max(1, min(int(total_steps), int(t)))
        alive_val = max(1, int(keep))
        y[t:] = float(alive_val)

    x = x_steps / float(total_steps)
    return x, y


def make_figure(*, profile_x_jitter_max: float, profile_y_jitter_max: float) -> plt.Figure:
    models = ["SD v1.5", "SDXL", "SD 3.5"]
    model_keys = ["sd15", "sdxl", "sd35"]
    model_total_steps = {"sd15": 64, "sdxl": 64, "sd35": 32}
    model_label_by_key = dict(zip(model_keys, models))
    metrics = ["IR", "HPS", "GenEval\n(IR)", "GenEval\n(HPS)"]

    # Regular-inference baseline values (3-seed mean on GenEval prompts), kept for reference.
    # fixed_mt = {
    #     "IR": [-0.1593, 0.4315, 1.0447],
    #     "HPS": [0.2567, 0.2748, 0.2968],
    #     "GenEval\n(IR)": [0.4340, 0.5289, 0.7130],
    #     "GenEval\n(HPS)": [0.4340, 0.5289, 0.7130],
    # }
    # PSP default schedule (k_init=8, ts=[16,32]/[8,16], ks=[4,2]); 3-seed mean on GenEval prompts.
    fixed_mt = {
        "IR": [0.8266, 1.2238, 1.3798],
        "HPS": [0.2877, 0.3035, 0.3158],
        "GenEval\n(IR)": [0.5737, 0.6449, 0.7466],
        "GenEval\n(HPS)": [0.5408, 0.6248, 0.7532],
    }
    fixed_tuned = {
        "IR": [0.8361, 1.2046, 1.3801],
        "HPS": [0.2880, 0.3050, 0.3159],
        "GenEval\n(IR)": [0.5710, 0.6405, 0.7525],
        "GenEval\n(HPS)": [0.5408, 0.6464, 0.7513],
    }

    # Keep model colors consistent across paper figures.
    model_colors = ["blue", "gold", "green"]

    fig = plt.figure(figsize=(12, 5.0))
    outer = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 1.25], wspace=0.30)
    left_gs = outer[0, 0].subgridspec(nrows=len(metrics), ncols=1, hspace=0.06)
    right_gs = outer[0, 1].subgridspec(nrows=len(model_keys), ncols=1, hspace=0.16)
    axes_left = [fig.add_subplot(left_gs[row, 0]) for row in range(len(metrics))]
    axes_right = [fig.add_subplot(right_gs[i, 0]) for i in range(len(model_keys))]
    axes_right_by_model = {k: ax for k, ax in zip(model_keys, axes_right)}
    x = np.arange(len(models))
    pair_offset = 0.18
    bar_width = 0.16

    for row, metric in enumerate(metrics):
        ax = axes_left[row]
        for i, color in enumerate(model_colors):
            ax.bar(
                x[i] - pair_offset,
                fixed_mt[metric][i],
                width=bar_width,
                color=color,
                edgecolor="black",
                linewidth=0.6,
            )
            ax.bar(
                x[i] + pair_offset,
                fixed_tuned[metric][i],
                width=bar_width,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                hatch="///",
            )

        if row == 0:
            ax.set_title("Effect of Tuning Pruning Schedule")

        ax.axhline(0, linewidth=0.8)
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", linewidth=0.6, alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))

    axes_left[-1].set_xticks(x)
    axes_left[-1].set_xticklabels(models)
    for ax in axes_left[:-1]:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axes_left[0].set_ylabel(metrics[0], labelpad=12)

    # Threshold for which schedules to draw on the right panel (independent of left-panel bars).
    # Uses regular-inference IR (GenEval prompts) so changing fixed_mt does not affect this filter.
    right_panel_baseline_ir = [-10.0, -10.0, -10.0]
    baseline_by_model = dict(zip(model_keys, right_panel_baseline_ir))
    repo_root = Path(__file__).resolve().parents[2]
    best_root = repo_root / "results" / "best_strategies"

    profiles_by_model: dict[str, list[dict[str, object]]] = {k: [] for k in model_keys}
    max_profiles_per_model: int | None = None
    max_items_to_scan: int | None = None

    for model in model_keys:
        model_path = best_root / model / "scheduled_ir_benchmark_ir.json"
        if not model_path.exists():
            continue

        collected = 0
        for item in _iter_strategy_items(model_path, max_items=max_items_to_scan):
            strategy = item.get("strategy", {})
            seed_mean = item.get("seed_mean", {})
            if not isinstance(strategy, dict) or not isinstance(seed_mean, dict):
                continue

            score = _safe_float(seed_mean.get("IR"))
            if not np.isfinite(score) or score <= baseline_by_model[model]:
                continue

            k_init = int(strategy.get("k_init", 0))
            cutoff_times = strategy.get("cutoff_times", [])
            remaining = strategy.get("remaining_particles", [])
            if not isinstance(cutoff_times, list) or not isinstance(remaining, list):
                continue
            if len(cutoff_times) != len(remaining):
                continue

            prof_x, prof_y = _profile_points(
                total_steps=model_total_steps[model],
                k_init=k_init,
                cutoff_times=[int(v) for v in cutoff_times],
                remaining_particles=[int(v) for v in remaining],
            )
            if prof_x.size == 0:
                continue

            profiles_by_model[model].append(
                {
                    "x": prof_x,
                    "y": prof_y,
                    "score": float(score),
                    "model": model,
                }
            )
            collected += 1
            if max_profiles_per_model is not None and collected >= max_profiles_per_model:
                break

    flat_profiles = [p for model in model_keys for p in profiles_by_model[model]]
    if flat_profiles:
        cmap = plt.get_cmap("viridis")
        rng = np.random.default_rng(7)

        for model in model_keys:
            model_profiles = profiles_by_model[model]
            if not model_profiles:
                continue
            ax = axes_right_by_model[model]
            model_scores = np.asarray([float(p["score"]) for p in model_profiles], dtype=np.float64)
            winning_score = float(np.max(model_scores))
            model_norm = plt.Normalize(
                vmin=float(np.min(model_scores)),
                vmax=float(np.max(model_scores)),
            )
            order = np.argsort([float(p["score"]) for p in model_profiles])

            for rank, local_draw_idx in enumerate(order):
                entry = model_profiles[int(local_draw_idx)]
                px = np.asarray(entry["x"], dtype=np.float64)
                py = np.asarray(entry["y"], dtype=np.float64)
                score_val = float(entry["score"])
                is_winner = np.isclose(score_val, winning_score, rtol=0.0, atol=1e-12)

                # Keep trajectory shape faithful: jitter as a small per-line offset.
                if is_winner:
                    x_add_jitter = 0.0
                    y_jitter = 0.0
                else:
                    x_add_jitter = float(rng.uniform(-profile_x_jitter_max, profile_x_jitter_max))
                    y_jitter = float(rng.uniform(-profile_y_jitter_max, profile_y_jitter_max))
                x_plot = np.clip(px + x_add_jitter, 0.0, 1.0)
                y_plot = np.maximum(1.0, py + y_jitter)
                color = "red" if is_winner else cmap(model_norm(score_val))
                line_width = 2.0 if is_winner else 1.0
                line_zorder = 20000 if is_winner else (2 + int(rank))
                ax.plot(
                    x_plot,
                    y_plot,
                    color=color,
                    linewidth=line_width,
                    alpha=0.65,
                    drawstyle="steps-post",
                    zorder=line_zorder,
                )

            # Reference fixed schedule: k_init=8, ts=[0.25T, 0.5T], ks=[4,2].
            total_steps = int(model_total_steps[model])
            ref_cutoffs = [max(1, int(round(0.25 * total_steps))), max(1, int(round(0.5 * total_steps)))]
            ref_x, ref_y = _profile_points(
                total_steps=total_steps,
                k_init=8,
                cutoff_times=ref_cutoffs,
                remaining_particles=[4, 2],
            )
            if ref_x.size > 0:
                ax.plot(
                    ref_x,
                    ref_y,
                    color="black",
                    linestyle=":",
                    linewidth=1.7,
                    alpha=0.95,
                    drawstyle="steps-post",
                    zorder=15000,
                )

            mappable = cm.ScalarMappable(norm=model_norm, cmap=cmap)
            cbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.01)
            cbar.set_label("Final IR")
            vmin = float(model_norm.vmin)
            vmax = float(model_norm.vmax)
            interp_fracs = np.array([0.10, 0.50, 0.90], dtype=np.float64)
            if np.isclose(vmin, vmax):
                tick_vals = np.array([vmin, vmin, vmin], dtype=np.float64)
            else:
                tick_vals = vmin + interp_fracs * (vmax - vmin)
            cbar.set_ticks(tick_vals.tolist())
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    for idx, model in enumerate(model_keys):
        ax = axes_right_by_model[model]
        ax.set_xscale("linear")
        ax.set_yscale("log", base=2)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1.0, 18.0)
        ax.set_yticks([1, 2, 4, 8, 16])
        ax.set_yticklabels(["1", "2", "4", "8", "16"])
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.grid(True, axis="both", linewidth=0.6, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == len(model_keys) - 1:
            ax.set_xlabel("inference progress (%)")
        else:
            ax.set_xticklabels([])
        ax.set_ylabel(f"{model_label_by_key[model]}\nParticles alive")

    # Shared title for the right panel stack.
    axes_right[0].set_title("Pruning Schedules Performance")

    model_handles = [
        Patch(facecolor=color, edgecolor="black", label=model)
        for model, color in zip(models, model_colors)
    ]
    style_handles = [
        Patch(facecolor="white", edgecolor="black", label="Default (solid)"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Tuned (hatched)"),
    ]
    # Center left legends around the SDXL x-tick in the left panel.
    sdxl_tick_fig_x = fig.transFigure.inverted().transform(
        axes_left[-1].transData.transform((1.0, 0.0))
    )[0]
    fig.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(sdxl_tick_fig_x, 0.090),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(sdxl_tick_fig_x, 0.040),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    right_handles = [
        Line2D([0], [0], color="red", linewidth=2.0, linestyle="-", label="Best schedule"),
        Line2D([0], [0], color="black", linewidth=1.7, linestyle=":", label="Default schedule"),
    ]
    right_x0 = min(ax.get_position().x0 for ax in axes_right)
    right_x1 = max(ax.get_position().x1 for ax in axes_right)
    right_center_x = 0.5 * (right_x0 + right_x1)
    fig.legend(
        handles=right_handles,
        loc="upper center",
        bbox_to_anchor=(right_center_x, 0.062),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.subplots_adjust(bottom=0.18)
    return fig


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = make_figure(
        profile_x_jitter_max=max(0.0, float(args.profile_x_jitter_max)),
        profile_y_jitter_max=max(0.0, float(args.profile_y_jitter_max)),
    )
    png_path = output_dir / f"{args.base_name}.png"
    pdf_path = output_dir / f"{args.base_name}.pdf"

    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
