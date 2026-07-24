#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from diffusers import DDIMScheduler
from tqdm import tqdm


@dataclass(frozen=True)
class ModelPreset:
    model_name: str
    default_steps: int
    model_tag: str


MODEL_PRESETS: Dict[str, ModelPreset] = {
    "sd15": ModelPreset("stable-diffusion-v1-5", 64, "sd15"),
    "sdxl": ModelPreset("stable-diffusion-xl", 64, "sdxl"),
    "sd35": ModelPreset("stable-diffusion-3.5-large", 32, "sd35"),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _setup_external_imports() -> None:
    repo_root = _repo_root()
    text_to_image = repo_root / "Fk-Diffusion-Steering" / "text_to_image"
    if str(text_to_image) not in sys.path:
        sys.path.insert(0, str(text_to_image))
    fkd_diffusers = text_to_image / "fkd_diffusers"
    if str(fkd_diffusers) not in sys.path:
        sys.path.insert(0, str(fkd_diffusers))


_setup_external_imports()

from collect_image_reward_signal import (  # noqa: E402
    build_pipeline,
    compute_x0_preds_non_sd35,
    compute_x0_preds_sd35,
    decode_latents_to_tensor_sd35,
)
from fkd_diffusers.fkd_pipeline_sd import latent_to_decode as latent_to_decode_sd  # noqa: E402
from fkd_diffusers.fkd_pipeline_sdxl import (  # noqa: E402
    FKDStableDiffusionXL,
    latent_to_decode as latent_to_decode_sdxl,
)
from fks_utils import do_eval  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DSearch-style guided sampling on Geneval prompts for SD1.5/SDXL/SD3.5 "
            "and emit FK-like outputs."
        )
    )
    parser.add_argument("--model-key", type=str, choices=list(MODEL_PRESETS.keys()), required=True)
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompts-path", type=str, default="")
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prompt-start-id", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--num-particles", type=int, default=4)
    parser.add_argument("--duplicate-size", type=int, default=1)
    parser.add_argument("--w", type=float, default=2.0)
    parser.add_argument("--oversamplerate", type=int, default=2)
    parser.add_argument("--search-schedule", type=str, default="exponential")
    parser.add_argument("--drop-schedule", type=str, default="exponential")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--guidance-reward-fn", type=str, choices=["ImageReward", "HumanPreference"], required=True)
    parser.add_argument("--metrics-to-compute", type=str, default="ImageReward#HumanPreference")
    parser.add_argument("--stochastic-sampling", action="store_true")
    parser.add_argument("--use-step-wrapper-stochastic", action="store_true")
    parser.add_argument("--gamma-target", type=float, default=None)
    parser.add_argument(
        "--show-step-progress",
        action="store_true",
        help="Show per-prompt per-step progress bar (useful for smoke tests).",
    )
    parser.add_argument(
        "--profile-time-breakdown",
        action="store_true",
        help=(
            "Profile prompt wall-time into approximate diffusion time vs reward-evaluation time "
            "(based on do_eval call timing)."
        ),
    )
    return parser.parse_args()


def _read_geneval_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(f"Expected list in {path}")
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def _search_probability(schedule: str, step_idx: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    if schedule == "all":
        return 1.0
    if schedule == "uniform":
        return 0.7
    frac = float(step_idx) / float(total_steps - 1)
    if schedule == "linear":
        return min(1.0, max(0.0, (step_idx + 10) / float(total_steps)))
    if schedule == "exponential":
        return float(1 - np.exp(-step_idx / max(1.0, (total_steps - 1) / 3.0)))
    return 1.0


def _keep_count(drop_schedule: str, step_idx: int, total_steps: int, base_particles: int, pop_size: int) -> int:
    if total_steps <= 1:
        return pop_size
    if drop_schedule == "none":
        return pop_size
    t = float(step_idx) / float(total_steps - 1)
    if drop_schedule == "linear":
        val = int(round(pop_size - t * (pop_size - base_particles)))
    elif drop_schedule == "quadratic":
        val = int(round(base_particles + (pop_size - base_particles) * (1 - t) ** 2))
    elif drop_schedule == "sigmoid":
        k = 10.0
        raw = pop_size / (1 + np.exp(-k * (t - 0.5)))
        val = int(round(raw))
    else:  # exponential default
        if pop_size == base_particles:
            val = pop_size
        else:
            val = int(round(pop_size * (base_particles / pop_size) ** t))
    return int(min(pop_size, max(base_particles, val)))


def _decode_non_sd35_x0(pipeline, x0_preds: torch.Tensor) -> torch.Tensor:
    decode_fn = latent_to_decode_sdxl if isinstance(pipeline, FKDStableDiffusionXL) else latent_to_decode_sd
    return decode_fn(model=pipeline, output_type="pil", latents=x0_preds).detach()


def _metric_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "result": [float(v) for v in arr.tolist()],
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
    }


def install_sd35_step_wrapper_stochastic(scheduler, gamma_target: float) -> None:
    original_step = scheduler.step
    gamma_cap = np.sqrt(2.0) - 1.0
    gamma_eff = float(np.clip(float(gamma_target), 0.0, gamma_cap))

    def wrapped_step(*step_args, **step_kwargs):
        out = original_step(*step_args, **step_kwargs)
        if gamma_eff <= 0.0:
            return out

        timestep = None
        if len(step_args) >= 2:
            timestep = step_args[1]
        elif "timestep" in step_kwargs:
            timestep = step_kwargs["timestep"]
        if timestep is None:
            return out

        if hasattr(scheduler, "index_for_timestep"):
            sigma_idx = scheduler.index_for_timestep(timestep)
        else:
            ts = scheduler.timesteps
            sigma_idx = int((ts == timestep).nonzero(as_tuple=True)[0][0].item())

        sigma = scheduler.sigmas[sigma_idx]
        sigma_f = float(sigma.item()) if isinstance(sigma, torch.Tensor) else float(sigma)
        noise_std = sigma_f * np.sqrt(max((1.0 + gamma_eff) ** 2 - 1.0, 0.0))
        if noise_std <= 0.0:
            return out

        prev_sample = out[0] if isinstance(out, tuple) else out.prev_sample
        generator = step_kwargs.get("generator", None)
        if isinstance(generator, torch.Generator):
            noise = torch.randn(
                prev_sample.shape,
                generator=generator,
                device=prev_sample.device,
                dtype=prev_sample.dtype,
            )
        else:
            noise = torch.randn_like(prev_sample)
        perturbed = prev_sample + noise_std * noise

        if isinstance(out, tuple):
            return (perturbed,) + out[1:]
        out.prev_sample = perturbed
        return out

    scheduler.step = wrapped_step


def _resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    preset = MODEL_PRESETS[args.model_key]
    if not args.model_name:
        args.model_name = preset.model_name
    if args.num_inference_steps is None:
        args.num_inference_steps = preset.default_steps
    if not args.prompts_path:
        args.prompts_path = str(
            _repo_root()
            / "Fk-Diffusion-Steering"
            / "text_to_image"
            / "prompt_files"
            / "geneval_metadata.jsonl"
        )
    if not args.output_root:
        args.output_root = str(_repo_root() / "results" / "dsearch")
    if args.model_key == "sd35":
        if not args.stochastic_sampling:
            args.stochastic_sampling = True
        if not args.use_step_wrapper_stochastic:
            args.use_step_wrapper_stochastic = True
        if args.gamma_target is None:
            args.gamma_target = 0.005
    return args


def _save_winner_files(exp_root: Path, seed: int, rows: List[dict]) -> None:
    csv_path = exp_root / f"dsearch_winner_geneval_seed{seed}.csv"
    jsonl_path = exp_root / f"dsearch_winner_geneval_seed{seed}.jsonl"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_id",
                "prompt",
                "seed",
                "guidance_reward_fn",
                "guidance_score",
                "image_reward",
                "human_preference",
                "sample_relpath",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    args = _resolve_defaults(parse_args())
    final_metrics_to_compute = [m for m in args.metrics_to_compute.split("#") if m]
    if "ImageReward" not in final_metrics_to_compute:
        final_metrics_to_compute.append("ImageReward")
    if "HumanPreference" not in final_metrics_to_compute:
        final_metrics_to_compute.append("HumanPreference")
    # For search-time guidance, only evaluate the active guidance metric at
    # intermediate states. Full metric evaluation happens once on final outputs.
    intermediate_metrics_to_compute = [args.guidance_reward_fn]

    prompts_path = Path(args.prompts_path).resolve()
    prompt_rows = _read_geneval_rows(prompts_path)
    if args.prompt_start_id < 0 or args.prompt_start_id >= len(prompt_rows):
        raise ValueError("--prompt-start-id out of range")

    remaining = len(prompt_rows) - args.prompt_start_id
    requested = remaining if args.num_prompts is None else args.num_prompts
    if requested < 1:
        raise ValueError("--num-prompts must be >= 1")
    selected_count = min(remaining, requested)
    prompt_end_id = args.prompt_start_id + selected_count
    selected_rows = prompt_rows[args.prompt_start_id:prompt_end_id]

    pop_size = int(args.num_particles) * int(args.oversamplerate)
    if pop_size < args.num_particles:
        raise ValueError("oversamplerate produced invalid pop_size")

    guidance_tag = "ir" if args.guidance_reward_fn == "ImageReward" else "hps"
    exp_name = f"dsearch_{args.model_key}_k4_t{args.num_inference_steps}_{guidance_tag}_geneval"
    exp_root = Path(args.output_root).resolve() / MODEL_PRESETS[args.model_key].model_tag / exp_name
    exp_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    seed_dir = exp_root / f"seed={args.seed}_{ts}"
    seed_dir.mkdir(parents=True, exist_ok=False)

    args_to_save = vars(args).copy()
    args_to_save["prompt_end_id"] = prompt_end_id
    args_to_save["resolved_num_prompts"] = selected_count
    args_to_save["population_size"] = pop_size
    (seed_dir / "args.json").write_text(json.dumps(args_to_save, indent=2), encoding="utf-8")

    pipeline, is_sd35 = build_pipeline(args.model_name, args.device)
    pipeline.set_progress_bar_config(disable=True)

    if not is_sd35:
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler.set_timesteps(args.num_inference_steps, device=pipeline.device)
    else:
        scheduler_cls = pipeline.scheduler.__class__
        scheduler_init_params = set(inspect.signature(scheduler_cls.__init__).parameters.keys())
        supports_stochastic_sampling = "stochastic_sampling" in scheduler_init_params
        if supports_stochastic_sampling:
            pipeline.scheduler = scheduler_cls.from_config(
                pipeline.scheduler.config,
                stochastic_sampling=(args.stochastic_sampling and not args.use_step_wrapper_stochastic),
            )
        else:
            pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config)
        if args.use_step_wrapper_stochastic:
            install_sd35_step_wrapper_stochastic(pipeline.scheduler, float(args.gamma_target))

    total_prompt_time = 0.0
    total_pipeline_wall_time = 0.0
    total_reward_eval_time = 0.0
    total_reward_eval_calls = 0
    n_prompts = 0
    agg_metrics: Dict[str, Dict[str, float]] = {
        metric: {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
        for metric in final_metrics_to_compute
    }
    winner_rows: List[dict] = []

    for local_idx, row in enumerate(tqdm(selected_rows, desc="Prompts", unit="prompt")):
        prompt_id = args.prompt_start_id + local_idx
        prompt_text = str(row.get("prompt", "")).strip()
        if not prompt_text:
            continue

        prompt_dir = seed_dir / f"{prompt_id:05d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "metadata.jsonl").write_text(json.dumps(row), encoding="utf-8")

        generators = [
            torch.Generator(device=args.device).manual_seed(args.seed + prompt_id * pop_size + i)
            for i in range(pop_size)
        ]
        prompt_list = [prompt_text] * pop_size
        search_rng = np.random.default_rng(args.seed + prompt_id)
        search_events = 0

        prompt_start = time.perf_counter()
        prompt_reward_eval_time = 0.0
        prompt_reward_eval_calls = 0
        prompt_pipeline_wall_time = 0.0
        step_pbar = (
            tqdm(
                total=int(args.num_inference_steps),
                desc=f"Prompt {prompt_id:05d} steps",
                unit="step",
                leave=False,
            )
            if args.show_step_progress
            else None
        )
        if is_sd35:
            prev_latents = pipeline.prepare_latents(
                batch_size=pop_size,
                num_channels_latents=pipeline.transformer.config.in_channels,
                height=pipeline.transformer.config.sample_size * pipeline.vae_scale_factor,
                width=pipeline.transformer.config.sample_size * pipeline.vae_scale_factor,
                dtype=pipeline.transformer.dtype,
                device=pipeline.device,
                generator=list(generators),
                latents=None,
            )
        else:
            prev_latents = pipeline.prepare_latents(
                batch_size=pop_size,
                num_channels_latents=pipeline.unet.config.in_channels,
                height=pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                width=pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                dtype=pipeline.unet.dtype,
                device=pipeline.device,
                generator=generators,
                latents=None,
            )

        final_metric_res: Dict[str, dict] = {}

        def _dsearch_callback(_pipeline, step_idx, t, callback_kwargs):
            nonlocal prev_latents, final_metric_res, search_events
            nonlocal prompt_reward_eval_time, prompt_reward_eval_calls
            if step_pbar is not None:
                step_pbar.update(1)

            latents = callback_kwargs["latents"]
            is_last = int(step_idx) >= int(args.num_inference_steps) - 1

            do_search = is_last
            if not is_last:
                p = _search_probability(args.search_schedule, int(step_idx), int(args.num_inference_steps))
                do_search = bool(search_rng.random() < p)

            if not do_search:
                prev_latents = latents
                return {"latents": latents}

            if is_sd35:
                x0_preds = compute_x0_preds_sd35(
                    pipeline=pipeline,
                    prev_latents=prev_latents,
                    t=t,
                    callback_kwargs=callback_kwargs,
                )
                image_tensor = decode_latents_to_tensor_sd35(pipeline=pipeline, latents=x0_preds).detach()
            else:
                x0_preds = compute_x0_preds_non_sd35(
                    pipeline=pipeline,
                    scheduler=pipeline.scheduler,
                    prev_latents=prev_latents,
                    t=t,
                    callback_kwargs=callback_kwargs,
                    eta=args.eta,
                )
                image_tensor = _decode_non_sd35_x0(pipeline, x0_preds)

            images = pipeline.image_processor.postprocess(image_tensor, output_type="pil")
            eval_start = time.perf_counter()
            metric_res = do_eval(
                prompt=prompt_list,
                images=images,
                metrics_to_compute=intermediate_metrics_to_compute,
            )
            eval_elapsed = time.perf_counter() - eval_start
            prompt_reward_eval_time += eval_elapsed
            prompt_reward_eval_calls += 1
            guidance_scores = np.asarray(metric_res[args.guidance_reward_fn]["result"], dtype=np.float64)
            final_metric_res = metric_res

            if not is_last:
                search_events += 1
                keep_n = _keep_count(
                    args.drop_schedule,
                    int(step_idx),
                    int(args.num_inference_steps),
                    int(args.num_particles),
                    int(pop_size),
                )
                order = np.argsort(guidance_scores)[::-1]
                source = order[:keep_n]
                # `w` controls selection sharpness: higher `w` favors top-scoring particles.
                source_scores = guidance_scores[source]
                temp = max(1e-6, 1.0 / max(1e-6, float(args.w)))
                logits = (source_scores - source_scores.max()) / temp
                probs = np.exp(logits)
                probs = probs / probs.sum()
                sampled_rel = search_rng.choice(len(source), size=len(source_scores) * args.oversamplerate, replace=True, p=probs)
                sampled = source[sampled_rel]
                sampled = sampled[:pop_size]
                latents = latents[sampled]
                # light jitter to keep stochastic exploration alive after resampling.
                jitter = torch.randn_like(latents) * (0.003 * float(args.w))
                latents = latents + jitter
                prev_latents = latents
                return {"latents": latents}

            prev_latents = latents
            return {"latents": latents}

        if is_sd35:
            callback_inputs = [
                "latents",
                "prompt_embeds",
                "negative_prompt_embeds",
                "pooled_prompt_embeds",
                "negative_pooled_prompt_embeds",
            ]
            with torch.no_grad():
                pipeline_start = time.perf_counter()
                output = pipeline(
                    prompt_list,
                    num_inference_steps=args.num_inference_steps,
                    generator=list(generators),
                    latents=prev_latents,
                    output_type="pil",
                    callback_on_step_end=_dsearch_callback,
                    callback_on_step_end_tensor_inputs=callback_inputs,
                )
                prompt_pipeline_wall_time += time.perf_counter() - pipeline_start
        else:
            callback_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
            if isinstance(pipeline, FKDStableDiffusionXL):
                callback_inputs.extend(
                    [
                        "add_text_embeds",
                        "negative_pooled_prompt_embeds",
                        "add_time_ids",
                        "negative_add_time_ids",
                    ]
                )
            with torch.no_grad():
                pipeline_start = time.perf_counter()
                output = pipeline(
                    prompt_list,
                    num_inference_steps=args.num_inference_steps,
                    eta=args.eta,
                    generator=generators,
                    latents=prev_latents,
                    output_type="pil",
                    callback_on_step_end=_dsearch_callback,
                    callback_on_step_end_tensor_inputs=callback_inputs,
                )
                prompt_pipeline_wall_time += time.perf_counter() - pipeline_start
        if step_pbar is not None:
            step_pbar.close()

        final_images = output.images if hasattr(output, "images") else output[0]
        # Always compute full metrics on final outputs, regardless of what was
        # computed at intermediate search steps.
        eval_start = time.perf_counter()
        final_metric_res = do_eval(
            prompt=prompt_list,
            images=final_images,
            metrics_to_compute=final_metrics_to_compute,
        )
        eval_elapsed = time.perf_counter() - eval_start
        prompt_reward_eval_time += eval_elapsed
        prompt_reward_eval_calls += 1

        prompt_elapsed = time.perf_counter() - prompt_start
        prompt_diffusion_approx_time = max(0.0, prompt_pipeline_wall_time - prompt_reward_eval_time)
        total_prompt_time += prompt_elapsed
        total_pipeline_wall_time += prompt_pipeline_wall_time
        total_reward_eval_time += prompt_reward_eval_time
        total_reward_eval_calls += prompt_reward_eval_calls
        n_prompts += 1

        guidance = np.asarray(final_metric_res[args.guidance_reward_fn]["result"], dtype=np.float64)
        order = np.argsort(guidance)[::-1]
        top = order[: args.num_particles]

        sorted_images = [final_images[i] for i in top]
        for metric in final_metrics_to_compute:
            metric_vals = np.asarray(final_metric_res[metric]["result"], dtype=np.float64)[top]
            final_metric_res[metric] = _metric_stats(metric_vals.tolist())
            agg_metrics[metric]["mean"] += final_metric_res[metric]["mean"]
            agg_metrics[metric]["max"] += final_metric_res[metric]["max"]
            agg_metrics[metric]["min"] += final_metric_res[metric]["min"]
            agg_metrics[metric]["std"] += final_metric_res[metric]["std"]

        final_metric_res["time_taken"] = float(prompt_elapsed)
        if args.profile_time_breakdown:
            final_metric_res["profile_pipeline_wall_time"] = float(prompt_pipeline_wall_time)
            final_metric_res["profile_reward_eval_time"] = float(prompt_reward_eval_time)
            final_metric_res["profile_diffusion_approx_time"] = float(prompt_diffusion_approx_time)
            final_metric_res["profile_reward_eval_calls"] = int(prompt_reward_eval_calls)
        final_metric_res["prompt"] = [prompt_text] * len(top)
        final_metric_res["prompt_index"] = int(prompt_id)
        final_metric_res["search_events"] = int(search_events)

        sample_dir = prompt_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for idx_img, image in enumerate(sorted_images):
            image.save(sample_dir / f"{idx_img:05d}.png")

        winner_dir = prompt_dir / "best_of_n_samples"
        winner_dir.mkdir(parents=True, exist_ok=True)
        if sorted_images:
            sorted_images[0].save(winner_dir / "00000.png")

        (prompt_dir / "results.json").write_text(json.dumps(final_metric_res), encoding="utf-8")

        winner_rows.append(
            {
                "prompt_id": int(prompt_id),
                "prompt": prompt_text,
                "seed": int(args.seed),
                "guidance_reward_fn": args.guidance_reward_fn,
                "guidance_score": float(final_metric_res[args.guidance_reward_fn]["result"][0]),
                "image_reward": float(final_metric_res["ImageReward"]["result"][0]),
                "human_preference": float(final_metric_res["HumanPreference"]["result"][0]),
                "sample_relpath": str(
                    Path(f"seed={args.seed}_{ts}") / f"{prompt_id:05d}" / "best_of_n_samples" / "00000.png"
                ),
            }
        )

    if n_prompts == 0:
        raise RuntimeError("No prompts were processed.")

    for metric in final_metrics_to_compute:
        for key in ["mean", "max", "min", "std"]:
            agg_metrics[metric][key] /= float(n_prompts)
    if args.profile_time_breakdown:
        agg_metrics["timing_profile"] = {
            "total_prompt_wall_time": float(total_prompt_time),
            "total_pipeline_wall_time": float(total_pipeline_wall_time),
            "total_reward_eval_time": float(total_reward_eval_time),
            "total_diffusion_approx_time": float(
                max(0.0, total_pipeline_wall_time - total_reward_eval_time)
            ),
            "reward_eval_calls": int(total_reward_eval_calls),
            "avg_prompt_wall_time": float(total_prompt_time / n_prompts),
            "avg_pipeline_wall_time": float(total_pipeline_wall_time / n_prompts),
            "avg_reward_eval_time": float(total_reward_eval_time / n_prompts),
            "avg_diffusion_approx_time": float(
                max(0.0, total_pipeline_wall_time - total_reward_eval_time) / n_prompts
            ),
        }
    (seed_dir / "final_metrics.json").write_text(json.dumps(agg_metrics), encoding="utf-8")

    _save_winner_files(exp_root, args.seed, winner_rows)
    print(f"Done. Output seed dir: {seed_dir}")


if __name__ == "__main__":
    main()
