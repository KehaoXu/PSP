#!/usr/bin/env python3
"""Re-score the chosen winner image of each prompt/seed with ImageReward.

This is a READ-ONLY utility: it never modifies any existing file. It only
*creates* a new summary .txt (and refuses to overwrite an existing one unless
--force is passed). It is meant for runs whose per-prompt results.json has an
empty ImageReward.result (i.e. IR was never scored), but whose winner images
under <prompt_id>/best_of_n_samples/ are all present.

For each logical seed it:
  * merges parallel shard folders (seed=<n>_<timestamp>),
  * dedupes by prompt_id (first sorted shard with a valid winner PNG wins),
  * scores the winner image with ImageReward-v1.0 (same loader the pipelines use),
  * reports the per-seed mean IR and the mean across seeds.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Run folder containing seed=<n>_<timestamp> shard dirs "
        "(e.g. results/fits_rbf_v1/sd15/fitsrbf_sd15_t64_ir_geneval).",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="rescored_ir_summary.txt",
        help="Filename for the NEW summary written inside --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for the new summary (default: --root).",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl",
        help="Fallback prompt text source (indexed by prompt id) if results.json lacks 'prompt'.",
    )
    parser.add_argument("--model-name", type=str, default="ImageReward-v1.0", help="ImageReward model name or checkpoint path.")
    parser.add_argument("--device", type=str, default="", help="Torch device (default: cuda if available else cpu).")
    parser.add_argument("--batch-size", type=int, default=16, help="Images scored per forward pass.")
    parser.add_argument("--winner-subdir", type=str, default="best_of_n_samples", help="Subdir holding the chosen winner PNG.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on prompts per seed for a quick smoke test (0 = all).")
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing report file.")
    return parser.parse_args()


def _setup_imports() -> None:
    """Make `image_reward_utils.rm_load` and the ImageReward package importable."""
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "Fk-Diffusion-Steering" / "text_to_image",
        repo_root / "Fk-Diffusion-Steering" / "text_to_image" / "fkd_diffusers",
        repo_root / "Fk-Diffusion-Steering" / "src" / "image-reward",  # local ImageReward fallback
    ]
    for path in candidates:
        p = str(path)
        if path.exists() and p not in sys.path:
            sys.path.insert(0, p)


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
    """Return [(prompt_id, prompt_text, winner_png_path)] deduped by prompt_id (first sorted shard wins)."""
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
            if not prompt_text:
                if 0 <= pid < len(metadata_rows):
                    prompt_text = metadata_rows[pid].get("prompt")
            if not prompt_text:
                raise ValueError(f"No prompt text for prompt_id={pid} ({prompt_dir}); pass a valid --metadata-path.")
            staged[pid] = (str(prompt_text), png)
    return [(pid, staged[pid][0], staged[pid][1]) for pid in sorted(staged)]


def main() -> None:
    args = parse_args()
    _setup_imports()

    import torch
    from image_reward_utils import rm_load

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--root does not exist: {root}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else root
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / args.report_name
    if report_path.exists() and not args.force:
        raise FileExistsError(f"Report already exists (use --force to overwrite): {report_path}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    metadata_rows = _read_metadata(Path(args.metadata_path).resolve())
    grouped = _group_shards_by_seed(root)
    if not grouped:
        raise ValueError(f"No seed=<n>_<timestamp> shard dirs under: {root}")

    print(f"Loading ImageReward model '{args.model_name}' on {device} ...")
    model = rm_load(args.model_name, device=device)
    model.eval()

    per_seed_rows: list[dict] = []
    for seed in sorted(grouped):
        entries = _collect_seed(grouped[seed], metadata_rows, args.winner_subdir)
        if args.limit > 0:
            entries = entries[: args.limit]
        if not entries:
            print(f"[seed={seed}] no winner images found, skipping.")
            continue

        scores: list[float] = []
        bs = max(1, int(args.batch_size))
        with torch.inference_mode():
            for start in tqdm(range(0, len(entries), bs), desc=f"seed={seed}", unit="batch"):
                batch = entries[start : start + bs]
                prompts = [e[1] for e in batch]
                images = [Image.open(e[2]).convert("RGB") for e in batch]
                if len(images) == 1:
                    scores.append(float(model.score(prompts[0], images[0])))
                else:
                    scores.extend(float(s) for s in model.score_batched(prompts, images))

        seed_mean = float(np.mean(scores))
        per_seed_rows.append({"seed": int(seed), "ir_mean": seed_mean, "num_prompts": len(scores)})
        print(f"[seed={seed}] IR_mean={seed_mean:.6f} over {len(scores)} prompts")

    if not per_seed_rows:
        raise RuntimeError(f"No IR scores computed for {root}")

    seed_means = np.asarray([r["ir_mean"] for r in per_seed_rows], dtype=np.float64)
    overall_mean = float(seed_means.mean())
    overall_std = float(seed_means.std(ddof=0))

    lines: list[str] = []
    lines.append(f"Re-scored IR (ImageReward) for: {root}")
    lines.append(f"Model: {args.model_name}   device: {device}")
    lines.append(f"Winner image: <prompt_id>/{args.winner_subdir}/<first .png>")
    lines.append("")
    lines.append("Per-seed results:")
    for r in per_seed_rows:
        lines.append(f"- seed={r['seed']}: IR_mean={r['ir_mean']:.6f}, prompts={r['num_prompts']}")
    lines.append("")
    lines.append(f"Mean across seeds: IR_mean={overall_mean:.6f} (std={overall_std:.6f})")
    text = "\n".join(lines)

    print("\n" + text)
    report_path.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
