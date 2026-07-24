#!/usr/bin/env python3
"""Plot VRAM-vs-time curves for BoN/PPS and PPS VAE decode chunk variants."""

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

BYTES_PER_GB = 1024 ** 3


def load_trace(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as h:
        return json.load(h)


def extract_method_curves(payload: dict, method: str) -> Tuple[List[float], List[float], List[float]]:
    rows = payload["averaged"][method]
    ts = [r["mean_t_s"] for r in rows]
    peak_gb = [r["mean_peak_alloc_bytes"] / BYTES_PER_GB for r in rows]
    std_gb = [r["std_peak_alloc_bytes"] / BYTES_PER_GB for r in rows]
    return ts, peak_gb, std_gb


def method_run_peak_gb(payload: dict, method: str) -> Tuple[float, float]:
    s = payload["peak_summary"][method]
    return s["mean_peak_alloc_bytes"] / BYTES_PER_GB, s["std_peak_alloc_bytes"] / BYTES_PER_GB


def method_wall_stats(payload: dict, method: str) -> Tuple[float, float, int]:
    stats_ids = set(payload.get("config", {}).get("stats_prompt_ids", []))
    rows = payload.get("per_prompt", {}).get(method, [])
    if stats_ids:
        rows = [r for r in rows if int(r.get("prompt_id", -1)) in stats_ids]
    vals = [float(r.get("total_wall_s", 0.0)) for r in rows]
    if not vals:
        return 0.0, 0.0, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return float(statistics.mean(vals)), float(statistics.stdev(vals)), len(vals)


def plot_panel(
    ax,
    base_payload: dict,
    vae1_payload: dict,
    vae2_payload: dict,
    vae4_payload: dict,
    title: str,
    *,
    colors: Dict[str, str],
) -> None:
    curves = [
        (base_payload, "bon", "BoN", colors["bon"]),
        (base_payload, "pps", "PPS", colors["pps"]),
        (vae1_payload, "pps", "PPS (VAE bs=1)", colors["vae1"]),
        (vae2_payload, "pps", "PPS (VAE bs=2)", colors["vae2"]),
        (vae4_payload, "pps", "PPS (VAE bs=4)", colors["vae4"]),
    ]
    for payload, method, label, color in curves:
        ts, peak_gb, _ = extract_method_curves(payload, method)
        ax.plot(ts, peak_gb, color=color, label=label, linewidth=1.6)
        run_peak_gb, _ = method_run_peak_gb(payload, method)
        ax.axhline(run_peak_gb, color=color, linestyle=":", linewidth=1.2, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("VRAM (GB)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=True)


def add_backbone_args(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument(f"--{prefix}-base", type=str, required=True)
    parser.add_argument(f"--{prefix}-vae1", type=str, required=True)
    parser.add_argument(f"--{prefix}-vae2", type=str, required=True)
    parser.add_argument(f"--{prefix}-vae4", type=str, required=True)


def print_runtime_summary(tag: str, base: dict, vae1: dict, vae2: dict, vae4: dict) -> None:
    print(f"[plot] {tag} runtime summary:")
    m, s, n = method_wall_stats(base, "bon")
    print(f"  BoN: mean={m:.3f}s std={s:.3f}s n={n}")
    m, s, n = method_wall_stats(base, "pps")
    print(f"  PPS: mean={m:.3f}s std={s:.3f}s n={n}")
    m, s, n = method_wall_stats(vae1, "pps")
    print(f"  PPS (VAE bs=1): mean={m:.3f}s std={s:.3f}s n={n}")
    m, s, n = method_wall_stats(vae2, "pps")
    print(f"  PPS (VAE bs=2): mean={m:.3f}s std={s:.3f}s n={n}")
    m, s, n = method_wall_stats(vae4, "pps")
    print(f"  PPS (VAE bs=4): mean={m:.3f}s std={s:.3f}s n={n}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot VRAM curves with BoN/PPS and PPS VAE-chunk variants on 3 backbones."
    )
    add_backbone_args(parser, "sd15")
    add_backbone_args(parser, "sdxl")
    add_backbone_args(parser, "sd35")
    parser.add_argument("--output", type=str, default="results/compute_estimates/rebuttal/vram_pps_vaechunk.pdf")
    parser.add_argument("--also-png", action="store_true")
    parser.add_argument("--width", type=float, default=15.0)
    parser.add_argument("--height", type=float, default=4.2)
    args = parser.parse_args()

    sd15_base = load_trace(Path(args.sd15_base))
    sd15_vae1 = load_trace(Path(args.sd15_vae1))
    sd15_vae2 = load_trace(Path(args.sd15_vae2))
    sd15_vae4 = load_trace(Path(args.sd15_vae4))

    sdxl_base = load_trace(Path(args.sdxl_base))
    sdxl_vae1 = load_trace(Path(args.sdxl_vae1))
    sdxl_vae2 = load_trace(Path(args.sdxl_vae2))
    sdxl_vae4 = load_trace(Path(args.sdxl_vae4))

    sd35_base = load_trace(Path(args.sd35_base))
    sd35_vae1 = load_trace(Path(args.sd35_vae1))
    sd35_vae2 = load_trace(Path(args.sd35_vae2))
    sd35_vae4 = load_trace(Path(args.sd35_vae4))

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {
        "bon": cycle[0],
        "pps": cycle[1],
        "vae1": cycle[2],
        "vae2": cycle[3],
        "vae4": cycle[4],
    }

    fig, axes = plt.subplots(1, 3, figsize=(args.width, args.height), constrained_layout=True)
    plot_panel(axes[0], sd15_base, sd15_vae1, sd15_vae2, sd15_vae4, "SD v1.5", colors=colors)
    plot_panel(axes[1], sdxl_base, sdxl_vae1, sdxl_vae2, sdxl_vae4, "SDXL", colors=colors)
    plot_panel(axes[2], sd35_base, sd35_vae1, sd35_vae2, sd35_vae4, "SD 3.5", colors=colors)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")
    if args.also_png or out_path.suffix.lower() == ".png":
        png_path = out_path.with_suffix(".png")
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        print(f"[plot] wrote {png_path}")

    print("")
    print_runtime_summary("SD v1.5", sd15_base, sd15_vae1, sd15_vae2, sd15_vae4)
    print_runtime_summary("SDXL", sdxl_base, sdxl_vae1, sdxl_vae2, sdxl_vae4)
    print_runtime_summary("SD 3.5", sd35_base, sd35_vae1, sd35_vae2, sd35_vae4)


if __name__ == "__main__":
    main()
