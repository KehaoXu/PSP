#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from tqdm import tqdm

@dataclass(frozen=True)
class ModelPreset:
    config_path: str
    steps: int
    guidance_scale: float
    image_size: int
    batch_size: int
    init_n_particles: int
    max_nfe: int
    model_tag: str
    sample_method: Optional[str]
    convert_scheduler: Optional[str]


MODEL_PRESETS: Dict[str, ModelPreset] = {
    "sd35": ModelPreset(
        config_path="config/compositional_image/rbf_sd35.yaml",
        steps=32,
        guidance_scale=7.0,
        image_size=1024,
        batch_size=2,
        init_n_particles=4,
        max_nfe=128,
        model_tag="sd35",
        sample_method="sde",
        convert_scheduler="vp",
    ),
    "sdxl": ModelPreset(
        config_path="config/compositional_image/rbf_sdxl.yaml",
        steps=64,
        guidance_scale=7.5,
        image_size=1024,
        batch_size=2,
        init_n_particles=8,
        max_nfe=256,
        model_tag="sdxl",
        sample_method="ode",
        convert_scheduler=None,
    ),
    "sd15": ModelPreset(
        config_path="config/compositional_image/rbf_sd15.yaml",
        steps=64,
        guidance_scale=7.5,
        image_size=512,
        batch_size=2,
        init_n_particles=8,
        max_nfe=256,
        model_tag="sd15",
        sample_method="ode",
        convert_scheduler=None,
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _fits_root() -> Path:
    return Path(__file__).resolve().parent


def _read_geneval_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected list in {path}")
        rows = payload
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def _save_winner_files(exp_root: Path, seed: int, rows: List[dict], shard_tag: str) -> None:
    if not rows:
        return
    csv_path = exp_root / f"fits_winner_geneval_seed{seed}_{shard_tag}.csv"
    jsonl_path = exp_root / f"fits_winner_geneval_seed{seed}_{shard_tag}.jsonl"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _load_imagereward_model():
    try:
        import ImageReward as RM
    except Exception as exc:  # pragma: no cover - import error surfaced to caller
        raise RuntimeError(
            "ImageReward is required to write per-prompt ImageReward.result. "
            "Install dependencies via FITS setup scripts."
        ) from exc
    return RM.load("ImageReward-v1.0")


def _score_imagereward(model, prompt_text: str, image_path: Path) -> float:
    with Image.open(image_path) as img:
        image = img.convert("RGB")
        score = model.score(prompt_text, [image])
    if isinstance(score, (list, tuple)):
        return float(score[0])
    try:
        return float(score[0])  # numpy/tensor-like
    except Exception:
        return float(score)


def _metric_payload(values: List[float]) -> Dict[str, object]:
    if not values:
        return {"result": [], "mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
    if len(values) == 1:
        v = float(values[0])
        return {"result": [v], "mean": v, "std": 0.0, "max": v, "min": v}
    mean_v = sum(values) / len(values)
    var_v = sum((v - mean_v) ** 2 for v in values) / len(values)
    return {
        "result": [float(v) for v in values],
        "mean": float(mean_v),
        "std": float(var_v ** 0.5),
        "max": float(max(values)),
        "min": float(min(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FITS RBF (ImageReward) on Geneval with DSearch/FK-compatible output folders."
    )
    parser.add_argument("--model-key", type=str, choices=list(MODEL_PRESETS.keys()), default="sd35")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt-start-id", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--prompts-path", type=str, default="")
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--config-path", type=str, default="")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--init-n-particles", type=int, default=None)
    parser.add_argument("--max-nfe", type=int, default=None)
    parser.add_argument("--ckpt-root", type=str, default="./ckpt")
    parser.add_argument("--reward-score", type=str, default="imagereward")
    parser.add_argument("--convert-scheduler", type=str, default="")
    parser.add_argument("--sample-method", type=str, default="")
    parser.add_argument("--filtering-method", type=str, default="rbf")
    return parser.parse_args()


def _resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    preset = MODEL_PRESETS[args.model_key]
    repo_root = _repo_root()
    fits_root = _fits_root()

    if not args.prompts_path:
        args.prompts_path = str(repo_root / "Fk-Diffusion-Steering" / "text_to_image" / "prompt_files" / "geneval_metadata.jsonl")
    if not args.output_root:
        args.output_root = str(repo_root / "results" / "fits_rbf_v1")
    if not args.config_path:
        args.config_path = str(fits_root / preset.config_path)
    if args.num_inference_steps is None:
        args.num_inference_steps = preset.steps
    if args.guidance_scale is None:
        args.guidance_scale = preset.guidance_scale
    if args.image_size is None:
        args.image_size = preset.image_size
    if args.batch_size is None:
        args.batch_size = preset.batch_size
    if args.init_n_particles is None:
        args.init_n_particles = preset.init_n_particles
    if args.max_nfe is None:
        args.max_nfe = preset.max_nfe
    if not args.sample_method:
        args.sample_method = preset.sample_method
    if not args.convert_scheduler:
        args.convert_scheduler = preset.convert_scheduler
    return args


def _locate_primary_output(run_root: Path) -> Path:
    candidates: List[Path] = []
    candidates.extend(run_root.rglob("output.png"))
    candidates.extend(run_root.rglob("output_*.png"))
    candidates.extend(run_root.rglob("output/*.png"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No output image found under {run_root}")
    candidates.sort(key=lambda p: (p.name, p.stat().st_mtime))
    return candidates[0]


def _run_fits_for_prompt(args: argparse.Namespace, prompt_text: str, sample_seed: int, run_root: Path) -> Dict[str, str]:
    run_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "main.py",
        "--config",
        str(Path(args.config_path).resolve()),
        f"text_prompt={prompt_text}",
        f"reward_score={args.reward_score}",
        f"filtering_method={args.filtering_method}",
        f"ckpt_root={args.ckpt_root}",
        f"batch_size={args.batch_size}",
        f"init_n_particles={args.init_n_particles}",
        f"max_nfe={args.max_nfe}",
        f"max_steps={args.num_inference_steps}",
        f"guidance_scale={args.guidance_scale}",
        f"height={args.image_size}",
        f"width={args.image_size}",
        f"seed={sample_seed}",
        f"root_dir={run_root.as_posix()}",
        "tag=run",
        "save_now=True",
        "disable_debug=True",
    ]
    if args.sample_method:
        cmd.append(f"sample_method={args.sample_method}")
    if args.convert_scheduler:
        cmd.append(f"convert_scheduler={args.convert_scheduler}")
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(_fits_root()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"FITS run failed for seed={sample_seed}\nCommand: {' '.join(cmd)}\nOutput:\n{proc.stdout}"
        )
    output_image = _locate_primary_output(run_root)
    return {
        "output_image": str(output_image),
        "wall_time_sec": f"{elapsed:.6f}",
    }


def main() -> None:
    args = _resolve_defaults(parse_args())
    imagereward_model = _load_imagereward_model() if str(args.reward_score).lower() == "imagereward" else None

    prompt_rows = _read_geneval_rows(Path(args.prompts_path).resolve())
    if args.prompt_start_id < 0 or args.prompt_start_id >= len(prompt_rows):
        raise ValueError("--prompt-start-id out of range")
    remaining = len(prompt_rows) - args.prompt_start_id
    requested = remaining if args.num_prompts is None else args.num_prompts
    selected_count = min(remaining, requested)
    selected_rows = prompt_rows[args.prompt_start_id : args.prompt_start_id + selected_count]
    if not selected_rows:
        raise ValueError("No prompts selected")

    preset = MODEL_PRESETS[args.model_key]
    exp_name = f"fitsrbf_{args.model_key}_t{args.num_inference_steps}_ir_geneval"
    exp_root = Path(args.output_root).resolve() / preset.model_tag / exp_name
    exp_root.mkdir(parents=True, exist_ok=True)
    seed_dir = exp_root / f"seed={args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    seed_dir.mkdir(parents=True, exist_ok=False)

    args_to_save = vars(args).copy()
    args_to_save["resolved_num_prompts"] = selected_count
    (seed_dir / "args.json").write_text(json.dumps(args_to_save, indent=2), encoding="utf-8")

    winner_rows: List[dict] = []
    audit_rows: List[dict] = []
    total_prompt_time = 0.0
    all_image_rewards: List[float] = []

    for local_idx, row in enumerate(tqdm(selected_rows, desc="Prompts", unit="prompt")):
        prompt_id = args.prompt_start_id + local_idx
        prompt_text = str(row.get("prompt", "")).strip()
        if not prompt_text:
            continue

        prompt_dir = seed_dir / f"{prompt_id:05d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "metadata.jsonl").write_text(json.dumps(row), encoding="utf-8")

        sample_seed = args.seed + prompt_id
        run_root = seed_dir / "_fits_runs" / f"prompt_{prompt_id:05d}"
        run_info = _run_fits_for_prompt(args, prompt_text, sample_seed, run_root)

        output_image_path = Path(run_info["output_image"])
        sample_dir = prompt_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_image_path, sample_dir / "00000.png")

        winner_dir = prompt_dir / "best_of_n_samples"
        winner_dir.mkdir(parents=True, exist_ok=True)
        winner_img_path = winner_dir / "00000.png"
        shutil.copy2(output_image_path, winner_img_path)

        prompt_wall_time = float(run_info["wall_time_sec"])
        total_prompt_time += prompt_wall_time
        prompt_ir_values: List[float] = []
        if imagereward_model is not None:
            ir_score = _score_imagereward(imagereward_model, prompt_text, winner_img_path)
            prompt_ir_values = [ir_score]
            all_image_rewards.append(ir_score)
        result_payload = {
            "ImageReward": _metric_payload(prompt_ir_values),
            "time_taken": prompt_wall_time,
            "prompt": [prompt_text],
            "prompt_index": int(prompt_id),
            "fits_config": {
                "model_key": args.model_key,
                "num_inference_steps": int(args.num_inference_steps),
                "guidance_scale": float(args.guidance_scale),
                "batch_size": int(args.batch_size),
                "init_n_particles": int(args.init_n_particles),
                "max_nfe": int(args.max_nfe),
                "sample_seed": int(sample_seed),
            },
        }
        (prompt_dir / "results.json").write_text(json.dumps(result_payload), encoding="utf-8")

        winner_rows.append(
            {
                "prompt_id": int(prompt_id),
                "prompt": prompt_text,
                "seed": int(args.seed),
                "guidance_reward_fn": "ImageReward",
                "guidance_score": float(prompt_ir_values[0]) if prompt_ir_values else "",
                "sample_relpath": str(Path(seed_dir.name) / f"{prompt_id:05d}" / "best_of_n_samples" / "00000.png"),
                "image_reward": float(prompt_ir_values[0]) if prompt_ir_values else "",
            }
        )
        audit_rows.append(
            {
                "prompt_id": int(prompt_id),
                "prompt": prompt_text,
                "prompt_wall_time": prompt_wall_time,
                "pipeline_wall_time": prompt_wall_time,
                "reward_eval_time": 0.0,
            }
        )
        # tqdm already provides prompt-level progress and ETA.

    if not winner_rows:
        raise RuntimeError("No prompts were processed.")

    avg_prompt_time = total_prompt_time / float(len(winner_rows))
    final_metrics = {
        "ImageReward": _metric_payload(all_image_rewards),
        "compute_audit": {
            "total_prompt_wall_time": total_prompt_time,
            "avg_prompt_wall_time": avg_prompt_time,
            "num_prompts": len(winner_rows),
            "num_inference_steps": int(args.num_inference_steps),
            "model_key": args.model_key,
        },
    }
    (seed_dir / "final_metrics.json").write_text(json.dumps(final_metrics), encoding="utf-8")

    with (seed_dir / "compute_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt_id", "prompt", "prompt_wall_time", "pipeline_wall_time", "reward_eval_time"],
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    shard_end = args.prompt_start_id + max(0, selected_count - 1)
    shard_tag = f"p{args.prompt_start_id:05d}-{shard_end:05d}"
    _save_winner_files(exp_root, args.seed, winner_rows, shard_tag)
    print(f"Done. Output seed dir: {seed_dir}")


if __name__ == "__main__":
    main()
