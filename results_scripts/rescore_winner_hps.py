#!/usr/bin/env python3
"""Score the chosen winner image of each prompt/seed with HPSv2 (v2.1).

Companion to rescore_winner_ir.py. Same READ-ONLY contract: it never modifies
any existing file; it only *creates* a new summary .txt (refusing to overwrite
unless --force). Intended to measure the human-preference score of the images
that were selected under ImageReward (IR) guidance.

For each logical seed it:
  * merges parallel shard folders (seed=<n>_<timestamp>),
  * dedupes by prompt_id (first sorted shard with a valid winner PNG wins),
  * scores the winner image with HPSv2 v2.1 (same call the pipelines use),
  * reports the per-seed mean HPS and the mean across seeds.

Works for any run laid out as <seed dir>/<prompt_id>/best_of_n_samples/<png>
(e.g. dsearch_v2, diffusion_tts_v2, fits_rbf_v1, bfs, svdd, fk_steering ir_geneval).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=str, required=True, help="Run folder containing seed=<n>_<timestamp> shard dirs.")
    parser.add_argument("--report-name", type=str, default="rescored_hps_summary.txt", help="Filename for the NEW summary written inside --output-dir.")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for the new summary (default: --root).")
    parser.add_argument("--metadata-path", type=str, default="Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl", help="Fallback prompt text source (indexed by prompt id).")
    parser.add_argument("--hps-version", type=str, default="v2.1", help="HPSv2 version string.")
    parser.add_argument("--winner-subdir", type=str, default="best_of_n_samples", help="Subdir holding the chosen winner PNG.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on prompts per seed for a quick smoke test (0 = all).")
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing report file.")
    return parser.parse_args()


def _install_turtle_shim_for_hpsv2() -> None:
    """hpsv2's open_clip does `from turtle import forward`; shim it out in headless envs."""
    if os.environ.get("FK_HPS_TURTLE_SHIM", "1").lower() not in {"1", "true", "yes", "on"}:
        return
    if "turtle" not in sys.modules:
        shim = types.ModuleType("turtle")
        shim.forward = lambda *_args, **_kwargs: None
        sys.modules["turtle"] = shim


def _read_metadata(metadata_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not metadata_path.exists():
        return rows
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_shards_by_seed(root: Path) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = {}
    for shard in sorted(root.glob("seed=*_*")):
        if not shard.is_dir():
            continue
        m = re.match(r"^seed=(\d+)_", shard.name)
        if not m:
            continue
        grouped.setdefault(int(m.group(1)), []).append(shard)
    return grouped


def _winner_png(prompt_dir: Path, winner_subdir: str) -> Path | None:
    wd = prompt_dir / winner_subdir
    if not wd.is_dir():
        return None
    pngs = sorted(p for p in wd.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    return pngs[0] if pngs else None


def _collect_seed(shard_dirs: list[Path], metadata_rows: list[dict], winner_subdir: str) -> list[tuple[int, str, Path]]:
    staged: dict[int, tuple[str, Path]] = {}
    for shard in sorted(shard_dirs):
        for prompt_dir in sorted((p for p in shard.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
            pid = int(prompt_dir.name)
            if pid in staged:
                continue
            png = _winner_png(prompt_dir, winner_subdir)
            if png is None:
                continue
            prompt_text = None
            results_path = prompt_dir / "results.json"
            if results_path.exists():
                try:
                    prompt_text = json.loads(results_path.read_text(encoding="utf-8")).get("prompt")
                except (ValueError, OSError):
                    prompt_text = None
            if not prompt_text and 0 <= pid < len(metadata_rows):
                prompt_text = metadata_rows[pid].get("prompt")
            if not prompt_text:
                raise ValueError(f"No prompt text for prompt_id={pid} ({prompt_dir}); pass a valid --metadata-path.")
            staged[pid] = (str(prompt_text), png)
    return [(pid, staged[pid][0], staged[pid][1]) for pid in sorted(staged)]


def main() -> None:
    args = parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--root does not exist: {root}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else root
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / args.report_name
    if report_path.exists() and not args.force:
        raise FileExistsError(f"Report already exists (use --force to overwrite): {report_path}")

    metadata_rows = _read_metadata(Path(args.metadata_path).resolve())
    grouped = _group_shards_by_seed(root)
    if not grouped:
        raise ValueError(f"No seed=<n>_<timestamp> shard dirs under: {root}")

    print("Loading HPSv2 ...")
    _install_turtle_shim_for_hpsv2()
    import hpsv2  # type: ignore

    def hps_score(image_path: Path, prompt: str) -> float:
        out = hpsv2.score(str(image_path), prompt, hps_version=args.hps_version)
        return float(out[0])

    per_seed_rows: list[dict] = []
    for seed in sorted(grouped):
        entries = _collect_seed(grouped[seed], metadata_rows, args.winner_subdir)
        if args.limit > 0:
            entries = entries[: args.limit]
        if not entries:
            print(f"[seed={seed}] no winner images found, skipping.")
            continue

        scores: list[float] = []
        for _pid, prompt, png in tqdm(entries, desc=f"seed={seed}", unit="img"):
            scores.append(hps_score(png, prompt))

        seed_mean = float(np.mean(scores))
        per_seed_rows.append({"seed": int(seed), "hps_mean": seed_mean, "num_prompts": len(scores)})
        print(f"[seed={seed}] HPS_mean={seed_mean:.6f} over {len(scores)} prompts")

    if not per_seed_rows:
        raise RuntimeError(f"No HPS scores computed for {root}")

    seed_means = np.asarray([r["hps_mean"] for r in per_seed_rows], dtype=np.float64)
    overall_mean = float(seed_means.mean())
    overall_std = float(seed_means.std(ddof=0))

    lines: list[str] = []
    lines.append(f"Re-scored HPS (HPSv2 {args.hps_version}) for: {root}")
    lines.append(f"Winner image: <prompt_id>/{args.winner_subdir}/<first .png>")
    lines.append("")
    lines.append("Per-seed results:")
    for r in per_seed_rows:
        lines.append(f"- seed={r['seed']}: HPS_mean={r['hps_mean']:.6f}, prompts={r['num_prompts']}")
    lines.append("")
    lines.append(f"Mean across seeds: HPS_mean={overall_mean:.6f} (std={overall_std:.6f})")
    text = "\n".join(lines)

    print("\n" + text)
    report_path.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
