#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DSearch outputs per logical seed using Geneval evaluate_images.py, "
            "while merging parallel shard folders."
        )
    )
    parser.add_argument(
        "--dsearch-root",
        type=str,
        required=True,
        help="DSearch experiment folder (e.g. results/dsearch/sd15/dsearch_sd15_k4_t64_ir_geneval).",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl",
        help="Geneval metadata jsonl indexed by prompt id.",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="dsearch_geneval_summary.txt",
        help="Output report filename saved inside --dsearch-root.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="",
        help="Optional model config path forwarded to evaluate_images.py.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="geneval/objdet",
        help="Model checkpoint directory forwarded to evaluate_images.py.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="Optional prompt cap per logical seed for quick smoke runs (0 means all prompts).",
    )
    return parser.parse_args()


def _read_metadata(metadata_path: Path) -> list[dict]:
    rows: list[dict] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No metadata rows found: {metadata_path}")
    return rows


def _logical_seed_from_name(name: str) -> int:
    m = re.match(r"^seed=(\d+)_", name)
    if not m:
        raise ValueError(f"Unexpected shard directory name: {name}")
    return int(m.group(1))


def _group_shards_by_seed(dsearch_root: Path) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = {}
    for shard_dir in sorted(dsearch_root.glob("seed=*_*")):
        if not shard_dir.is_dir():
            continue
        seed = _logical_seed_from_name(shard_dir.name)
        grouped.setdefault(seed, []).append(shard_dir)
    if not grouped:
        raise ValueError(f"No shard dirs matching seed=*_* found under: {dsearch_root}")
    return grouped


def _guidance_metric(folder_name: str, guidance_reward_fn: str) -> tuple[str, str]:
    low = f"{folder_name}::{guidance_reward_fn}".lower()
    if "humanpreference" in low or "hps" in low:
        return "HumanPreference", "HPS"
    return "ImageReward", "IR"


def _prompt_dirs(shard_dir: Path) -> list[Path]:
    return sorted(
        [p for p in shard_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )


def _geneval_overall(eval_jsonl: Path) -> float:
    df = pd.read_json(eval_jsonl, orient="records", lines=True)
    if df.empty:
        raise ValueError(f"Empty evaluation output: {eval_jsonl}")
    if "tag" not in df.columns or "correct" not in df.columns:
        raise ValueError(f"Missing required columns in eval output: {eval_jsonl}")
    task_scores = [float(task_df["correct"].mean()) for _, task_df in df.groupby("tag", sort=False)]
    if not task_scores:
        raise ValueError(f"No task scores computed from: {eval_jsonl}")
    return float(np.mean(task_scores))


def _run_evaluate_images(
    imagedir: Path,
    outfile: Path,
    *,
    model_path: str,
    model_config: str,
) -> None:
    cmd = [
        "python",
        "geneval/evaluation/evaluate_images.py",
        str(imagedir),
        "--outfile",
        str(outfile),
        "--samples-dir",
        "samples",
        "--model-path",
        model_path,
    ]
    if model_config:
        cmd.extend(["--model-config", model_config])
    subprocess.run(cmd, check=True)


def _collect_seed_prompts(
    seed: int,
    shard_dirs: list[Path],
    metadata_rows: list[dict],
    *,
    folder_name: str,
    max_prompts: int,
) -> tuple[Path, list[float], str, str, int]:
    prompt_to_stage: dict[int, tuple[Path, float]] = {}
    metric_key: str | None = None
    metric_label: str | None = None

    for shard_dir in sorted(shard_dirs):
        args_path = shard_dir / "args.json"
        if not args_path.exists():
            continue
        args_obj = json.loads(args_path.read_text(encoding="utf-8"))
        guidance_reward_fn = str(args_obj.get("guidance_reward_fn", ""))
        cur_key, cur_label = _guidance_metric(folder_name, guidance_reward_fn)
        if metric_key is None:
            metric_key = cur_key
            metric_label = cur_label
        for prompt_dir in _prompt_dirs(shard_dir):
            prompt_id = int(prompt_dir.name)
            if prompt_id in prompt_to_stage:
                continue
            if prompt_id < 0 or prompt_id >= len(metadata_rows):
                continue

            results_path = prompt_dir / "results.json"
            winners_dir = prompt_dir / "best_of_n_samples"
            if not results_path.exists() or not winners_dir.exists():
                continue

            result_obj = json.loads(results_path.read_text(encoding="utf-8"))
            metric_obj = result_obj.get(metric_key, {})
            metric_arr = metric_obj.get("result")
            if isinstance(metric_arr, list) and metric_arr:
                guidance_value = float(max(metric_arr))
            else:
                metric_mean = metric_obj.get("mean", 0.0)
                try:
                    guidance_value = float(metric_mean)
                except (TypeError, ValueError):
                    guidance_value = 0.0

            winner_imgs = sorted(
                [p for p in winners_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
            )
            if not winner_imgs:
                continue

            prompt_to_stage[prompt_id] = (prompt_dir, guidance_value)

    if metric_key is None or metric_label is None:
        raise ValueError(f"Could not infer guidance metric for seed={seed}")
    if not prompt_to_stage:
        raise ValueError(f"No valid prompts with winners found for logical seed={seed}")

    tmp_root = Path(tempfile.mkdtemp(prefix=f"dsearch_eval_seed{seed}_"))
    guidance_values: list[float] = []
    prompt_ids = sorted(prompt_to_stage.keys())
    if max_prompts > 0:
        prompt_ids = prompt_ids[:max_prompts]

    valid_prompts = 0
    for prompt_id in prompt_ids:
        prompt_dir, guidance_value = prompt_to_stage[prompt_id]
        stage_prompt = tmp_root / f"{prompt_id:05d}"
        stage_samples = stage_prompt / "samples"
        stage_samples.mkdir(parents=True, exist_ok=True)

        winners_dir = prompt_dir / "best_of_n_samples"
        winner_imgs = sorted([p for p in winners_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
        for src_img in winner_imgs:
            dst_img = stage_samples / src_img.name
            try:
                os.symlink(src_img.resolve(), dst_img)
            except OSError:
                shutil.copy2(src_img, dst_img)

        (stage_prompt / "metadata.jsonl").write_text(
            json.dumps(metadata_rows[prompt_id]),
            encoding="utf-8",
        )
        guidance_values.append(guidance_value)
        valid_prompts += 1

    if valid_prompts == 0:
        raise ValueError(f"All prompts filtered out for logical seed={seed}")
    return tmp_root, guidance_values, metric_key, metric_label, valid_prompts


def main() -> None:
    args = parse_args()
    dsearch_root = Path(args.dsearch_root).resolve()
    metadata_path = Path(args.metadata_path).resolve()

    if not dsearch_root.exists():
        raise FileNotFoundError(f"--dsearch-root does not exist: {dsearch_root}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"--metadata-path does not exist: {metadata_path}")

    metadata_rows = _read_metadata(metadata_path)
    grouped = _group_shards_by_seed(dsearch_root)

    per_seed_rows: list[dict] = []
    guidance_label = "IR"
    for seed, shard_dirs in sorted(grouped.items()):
        tmp_root, guidance_values, _, metric_label, valid_prompts = _collect_seed_prompts(
            seed,
            shard_dirs,
            metadata_rows,
            folder_name=dsearch_root.name,
            max_prompts=args.max_prompts,
        )
        guidance_label = metric_label
        eval_out = tmp_root / "eval_results.jsonl"
        try:
            _run_evaluate_images(
                tmp_root,
                eval_out,
                model_path=args.model_path,
                model_config=args.model_config,
            )
            geneval = _geneval_overall(eval_out)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        per_seed_rows.append(
            {
                "seed": int(seed),
                "guidance_mean": float(np.mean(guidance_values)),
                "geneval_overall": float(geneval),
                "num_prompts": int(valid_prompts),
                "num_shards": int(len(shard_dirs)),
            }
        )

    if not per_seed_rows:
        raise RuntimeError(f"No seed results computed for {dsearch_root}")

    df = pd.DataFrame(per_seed_rows).sort_values("seed")
    g_mean = float(df["guidance_mean"].mean())
    g_std = float(df["guidance_mean"].std(ddof=0))
    ge_mean = float(df["geneval_overall"].mean())
    ge_std = float(df["geneval_overall"].std(ddof=0))

    lines = []
    lines.append(f"DSearch root: {dsearch_root}")
    lines.append(f"Guidance metric: {guidance_label}")
    lines.append("")
    lines.append("Per-seed results:")
    for _, row in df.iterrows():
        lines.append(
            f"- seed={int(row['seed'])}: {guidance_label}_mean={row['guidance_mean']:.6f}, "
            f"GenEval={row['geneval_overall']:.6f}, prompts={int(row['num_prompts'])}, "
            f"shards={int(row['num_shards'])}"
        )
    lines.append("")
    lines.append(
        f"Mean across seeds: {guidance_label}_mean={g_mean:.6f} (std={g_std:.6f}), "
        f"GenEval={ge_mean:.6f} (std={ge_std:.6f})"
    )

    text = "\n".join(lines)
    print(text)

    report_path = dsearch_root / args.report_name
    report_path.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
