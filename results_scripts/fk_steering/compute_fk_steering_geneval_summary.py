#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FK-steering outputs per seed using Geneval evaluate_images.py logic, "
            "and summarize guidance + Geneval scores."
        )
    )
    parser.add_argument(
        "--fk-root",
        type=str,
        required=True,
        help="FK experiment folder (e.g. results/fk_steering/sd15/fk_sdv15_k4_t64_eta1_ir_geneval).",
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
        default="fk_geneval_summary.txt",
        help="Output report filename saved inside --fk-root.",
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
        help="Optional prompt cap per seed for quick smoke runs (0 means all prompts).",
    )
    return parser.parse_args()


def _read_metadata(metadata_path: Path) -> list[dict]:
    rows = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No metadata rows found: {metadata_path}")
    return rows


def _seed_dirs(fk_root: Path) -> list[Path]:
    out = []
    for p in sorted(fk_root.glob("seed=*_*")):
        if p.is_dir():
            out.append(p)
    if not out:
        raise ValueError(f"No seed dirs found under: {fk_root}")
    return out


def _guidance_metric(folder_name: str, args_json: dict) -> tuple[str, str]:
    fn = str(args_json.get("guidance_reward_fn", "")).lower()
    if "humanpreference" in fn or "hps" in folder_name.lower():
        return "HumanPreference", "HPS"
    return "ImageReward", "IR"


def _prompt_dirs(seed_dir: Path) -> list[Path]:
    return sorted([p for p in seed_dir.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))


def _build_eval_staging(
    seed_dir: Path,
    metadata_rows: list[dict],
    *,
    max_prompts: int,
) -> tuple[Path, list[float], str, str, int]:
    args_json = json.loads((seed_dir / "args.json").read_text(encoding="utf-8"))
    metric_key, metric_label = _guidance_metric(seed_dir.parent.name, args_json)
    prompt_dirs = _prompt_dirs(seed_dir)
    if max_prompts > 0:
        prompt_dirs = prompt_dirs[:max_prompts]

    if not prompt_dirs:
        raise ValueError(f"No prompt directories found in {seed_dir}")

    tmp_root = Path(tempfile.mkdtemp(prefix=f"fk_eval_{seed_dir.name}_"))
    guidance_values: list[float] = []
    valid_prompts = 0

    for prompt_dir in prompt_dirs:
        prompt_id = int(prompt_dir.name)
        if prompt_id < 0 or prompt_id >= len(metadata_rows):
            raise ValueError(f"Prompt id {prompt_id} out of metadata range in {seed_dir}")

        results_path = prompt_dir / "results.json"
        winners_dir = prompt_dir / "best_of_n_samples"
        if not results_path.exists() or not winners_dir.exists():
            continue

        result_obj = json.loads(results_path.read_text(encoding="utf-8"))
        metric_arr = result_obj.get(metric_key, {}).get("result", None)
        if not isinstance(metric_arr, list) or not metric_arr:
            continue

        # FK winner is the top guidance particle at the final step.
        guidance_values.append(float(max(metric_arr)))

        winner_imgs = sorted([p for p in winners_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
        if not winner_imgs:
            continue

        stage_prompt = tmp_root / prompt_dir.name
        stage_samples = stage_prompt / "samples"
        stage_samples.mkdir(parents=True, exist_ok=True)
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
        valid_prompts += 1

    if valid_prompts == 0:
        raise ValueError(f"No valid prompts with winners found for {seed_dir}")
    return tmp_root, guidance_values, metric_key, metric_label, valid_prompts


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


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    fk_root = Path(args.fk_root).resolve()
    metadata_path = Path(args.metadata_path).resolve()

    if not fk_root.exists():
        raise FileNotFoundError(f"--fk-root does not exist: {fk_root}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"--metadata-path does not exist: {metadata_path}")

    metadata_rows = _read_metadata(metadata_path)
    seed_dirs = _seed_dirs(fk_root)

    per_seed_rows: list[dict] = []
    guidance_label = "IR"

    for seed_dir in seed_dirs:
        tmp_root, guidance_values, _, metric_label, valid_prompts = _build_eval_staging(
            seed_dir,
            metadata_rows,
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

        seed_id = seed_dir.name.split("_")[0].replace("seed=", "")
        guidance_mean = float(np.mean(guidance_values))
        per_seed_rows.append(
            {
                "seed": seed_id,
                "guidance_mean": guidance_mean,
                "geneval_overall": geneval,
                "num_prompts": valid_prompts,
            }
        )

    if not per_seed_rows:
        raise RuntimeError(f"No seed results computed for {fk_root}")

    df = pd.DataFrame(per_seed_rows).sort_values("seed")
    g_mean = float(df["guidance_mean"].mean())
    g_std = float(df["guidance_mean"].std(ddof=0))
    ge_mean = float(df["geneval_overall"].mean())
    ge_std = float(df["geneval_overall"].std(ddof=0))

    lines = []
    lines.append(f"FK root: {fk_root}")
    lines.append(f"Guidance metric: {guidance_label}")
    lines.append("")
    lines.append("Per-seed results:")
    for _, row in df.iterrows():
        lines.append(
            f"- seed={row['seed']}: {guidance_label}_mean={row['guidance_mean']:.6f}, "
            f"GenEval={row['geneval_overall']:.6f}, prompts={int(row['num_prompts'])}"
        )
    lines.append("")
    lines.append(
        f"Mean across seeds: {guidance_label}_mean={g_mean:.6f} (std={g_std:.6f}), "
        f"GenEval={ge_mean:.6f} (std={ge_std:.6f})"
    )

    text = "\n".join(lines)
    print(text)

    report_path = fk_root / args.report_name
    report_path.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
