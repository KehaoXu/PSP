#!/usr/bin/env python3
"""Batch HPSv2 (v2.1) scoring of IR-guidance winner images, with a single
higher-level progress bar/ETA spanning all selected methods + backbones.

READ-ONLY contract: never modifies any existing file; only *creates* a new
per-run summary .txt (refusing to overwrite unless --force).

Selection:
  --group 1    -> dsearch, diffusion_tts, fits_rbf   (all backbones)
  --group 2    -> bfs, svdd, fk_steering             (all backbones)
  --group all  -> all six methods                    (default)
  --methods dsearch svdd ...   -> explicit subset (overrides --group)

Each run folder is <seed dir>/<prompt_id>/best_of_n_samples/<png>. For every
selected (method, backbone) the script writes rescored_hps_summary.txt with the
per-seed mean HPS and the mean across seeds, and prints the same to the terminal.
The top-level tqdm advances one step per scored image over the whole workload,
so the ETA reflects everything the command will do.
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


# method -> {backbone: run folder}
METHOD_ROOTS: dict[str, dict[str, str]] = {
    "dsearch": {
        "sd15": "results/dsearch_v2/sd15/dsearch_sd15_k4_t64_ir_geneval",
        "sdxl": "results/dsearch_v2/sdxl/dsearch_sdxl_k4_t64_ir_geneval",
        "sd35": "results/dsearch_v2/sd35/dsearch_sd35_k4_t32_ir_geneval",
    },
    "diffusion_tts": {
        "sd15": "results/diffusion_tts_v2/sd15/epsgreedy_sd15_t64_ir_geneval",
        "sdxl": "results/diffusion_tts_v2/sdxl/epsgreedy_sdxl_t64_ir_geneval",
        "sd35": "results/diffusion_tts_v2/sd35/epsgreedy_sd35_t32_ir_geneval",
    },
    "fits_rbf": {
        "sd15": "results/fits_rbf_v1/sd15/fitsrbf_sd15_t64_ir_geneval",
        "sdxl": "results/fits_rbf_v1/sdxl/fitsrbf_sdxl_t64_ir_geneval",
        "sd35": "results/fits_rbf_v1/sd35/fitsrbf_sd35_t32_ir_geneval",
    },
    "bfs": {
        "sd15": "results/bfs/sd15/bfs_sdv15_t64_ir_geneval",
        "sdxl": "results/bfs/sdxl/bfs_sdxl_t64_ir_geneval",
        "sd35": "results/bfs/sd35/bfs_sdv35_t32_ir_geneval",
    },
    "svdd": {
        "sd15": "results/svdd/sd15/svdd_sd15_t64_ir_geneval",
        "sdxl": "results/svdd/sdxl/svdd_sdxl_t64_ir_geneval",
        "sd35": "results/svdd/sd35/svdd_sd35_t32_ir_geneval",
    },
    "fk_steering": {
        "sd15": "results/fk_steering/sd15/fk_sdv15_k4_t64_eta1_ir_geneval",
        "sdxl": "results/fk_steering/sdxl/fk_sdxl_k4_t64_eta1_ir_geneval",
        "sd35": "results/fk_steering/sd35/fk_sdv35_k4_t32_gamma_0p005_ir_geneval",
    },
}

GROUPS: dict[str, list[str]] = {
    "1": ["dsearch", "diffusion_tts", "fits_rbf"],
    "2": ["bfs", "svdd", "fk_steering"],
    "all": list(METHOD_ROOTS.keys()),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", type=str, default="all", choices=sorted(GROUPS.keys()), help="Preset method group (default: all).")
    parser.add_argument("--methods", type=str, nargs="+", default=None, choices=sorted(METHOD_ROOTS.keys()), help="Explicit method subset (overrides --group).")
    parser.add_argument("--report-name", type=str, default="rescored_hps_summary.txt", help="Per-run summary filename.")
    parser.add_argument("--metadata-path", type=str, default="Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl", help="Fallback prompt text source (indexed by prompt id).")
    parser.add_argument("--hps-version", type=str, default="v2.1", help="HPSv2 version string.")
    parser.add_argument("--batch-size", type=int, default=32, help="Images per batched forward (model/checkpoint loaded once).")
    parser.add_argument("--winner-subdir", type=str, default="best_of_n_samples", help="Subdir holding the chosen winner PNG.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on prompts per seed for a quick smoke test (0 = all).")
    parser.add_argument("--force", action="store_true", help="Allow overwriting existing report files.")
    return parser.parse_args()


def _install_turtle_shim_for_hpsv2() -> None:
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
    methods = args.methods if args.methods else GROUPS[args.group]

    metadata_rows = _read_metadata(Path(args.metadata_path).resolve())

    # Build the full plan first so the top-level bar has an exact total.
    plan: list[dict] = []  # {method, backbone, root, report_path, seed_entries: {seed: entries}}
    total_images = 0
    for method in methods:
        for backbone, rel in METHOD_ROOTS[method].items():
            root = Path(rel).resolve()
            if not root.is_dir():
                print(f"[skip] missing run dir: {root}")
                continue
            report_path = root / args.report_name
            if report_path.exists() and not args.force:
                raise FileExistsError(f"Report already exists (use --force): {report_path}")
            grouped = _group_shards_by_seed(root)
            if not grouped:
                print(f"[skip] no seed dirs under: {root}")
                continue
            seed_entries: dict[int, list[tuple[int, str, Path]]] = {}
            for seed in sorted(grouped):
                entries = _collect_seed(grouped[seed], metadata_rows, args.winner_subdir)
                if args.limit > 0:
                    entries = entries[: args.limit]
                seed_entries[seed] = entries
                total_images += len(entries)
            plan.append({"method": method, "backbone": backbone, "root": root, "report_path": report_path, "seed_entries": seed_entries})

    if not plan or total_images == 0:
        raise RuntimeError("Nothing to score for the selected methods.")

    print(f"Selected methods: {methods}")
    print(f"Runs: {len(plan)}   Total images to score: {total_images}")
    print("Loading HPSv2 (once) ...")
    _install_turtle_shim_for_hpsv2()

    import torch
    import huggingface_hub
    from PIL import Image
    import hpsv2.img_score as hps_img  # type: ignore
    from hpsv2.src.open_clip import get_tokenizer  # type: ignore
    from hpsv2.utils import hps_version_map  # type: ignore

    # Load model + checkpoint + tokenizer exactly ONCE (hpsv2.score reloads these per image).
    hps_img.initialize_model()
    model = hps_img.model_dict["model"]
    preprocess_val = hps_img.model_dict["preprocess_val"]
    device = hps_img.device
    cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[args.hps_version])
    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    tokenizer = get_tokenizer("ViT-H-14")
    model = model.to(device)
    model.eval()
    del checkpoint

    def hps_score_batch(image_paths: list[Path], prompts: list[str]) -> list[float]:
        images = torch.stack([preprocess_val(Image.open(p).convert("RGB")) for p in image_paths]).to(device, non_blocking=True)
        text = tokenizer(prompts).to(device, non_blocking=True)
        with torch.inference_mode(), torch.cuda.amp.autocast():
            outputs = model(images, text)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            # HPSv2 score = diagonal of image_features @ text_features.T (per image vs its own prompt).
            per_sample = (image_features * text_features).sum(dim=-1)
        return [float(v) for v in per_sample.detach().float().cpu().numpy()]

    bs = max(1, int(args.batch_size))
    pbar = tqdm(total=total_images, desc="HPS", unit="img")
    for run in plan:
        method, backbone, root, report_path = run["method"], run["backbone"], run["root"], run["report_path"]
        per_seed_rows: list[dict] = []
        for seed in sorted(run["seed_entries"]):
            entries = run["seed_entries"][seed]
            scores: list[float] = []
            for start in range(0, len(entries), bs):
                chunk = entries[start : start + bs]
                pbar.set_postfix(method=method, bb=backbone, seed=seed)
                scores.extend(hps_score_batch([e[2] for e in chunk], [e[1] for e in chunk]))
                pbar.update(len(chunk))
            if scores:
                seed_mean = float(np.mean(scores))
                per_seed_rows.append({"seed": int(seed), "hps_mean": seed_mean, "num_prompts": len(scores)})

        if not per_seed_rows:
            continue
        seed_means = np.asarray([r["hps_mean"] for r in per_seed_rows], dtype=np.float64)
        overall_mean = float(seed_means.mean())
        overall_std = float(seed_means.std(ddof=0))

        lines = [
            f"Re-scored HPS (HPSv2 {args.hps_version}) for: {root}",
            f"Method: {method}   Backbone: {backbone}",
            f"Winner image: <prompt_id>/{args.winner_subdir}/<first .png>",
            "",
            "Per-seed results:",
        ]
        for r in per_seed_rows:
            lines.append(f"- seed={r['seed']}: HPS_mean={r['hps_mean']:.6f}, prompts={r['num_prompts']}")
        lines.append("")
        lines.append(f"Mean across seeds: HPS_mean={overall_mean:.6f} (std={overall_std:.6f})")
        text = "\n".join(lines)
        report_path.write_text(text + "\n", encoding="utf-8")
        pbar.write(f"[{method}/{backbone}] HPS_mean={overall_mean:.6f} -> {report_path}")

    pbar.close()
    print("Done.")


if __name__ == "__main__":
    main()
