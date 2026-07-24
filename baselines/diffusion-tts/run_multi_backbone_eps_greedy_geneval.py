#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
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
    guidance_scale: float
    model_tag: str


MODEL_PRESETS: Dict[str, ModelPreset] = {
    "sd15": ModelPreset("stable-diffusion-v1-5", 64, 7.5, "sd15"),
    "sdxl": ModelPreset("stable-diffusion-xl", 64, 7.5, "sdxl"),
    "sd35": ModelPreset("stable-diffusion-3.5-large", 32, 7.0, "sd35"),
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


def _import_sd_image_reward_scorer():
    sd_dir = Path(__file__).resolve().parent / "sd"
    if str(sd_dir) not in sys.path:
        sys.path.insert(0, str(sd_dir))
    from scorers import ImageRewardScorer  # type: ignore

    return ImageRewardScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run diffusion-tts eps-greedy on Geneval with DSearch-compatible outputs for sd15/sdxl/sd35."
    )
    parser.add_argument("--model-key", type=str, choices=list(MODEL_PRESETS.keys()), default="sd15")
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prompt-start-id", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--prompts-path", type=str, default="")
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--method", type=str, choices=["eps_greedy", "zero_order"], default="eps_greedy")
    parser.add_argument("--guidance-reward-fn", type=str, choices=["ImageReward"], default="ImageReward")
    parser.add_argument("--metrics-to-compute", type=str, default="ImageReward")

    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--bs", type=int, default=1)

    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--lambda_", type=float, default=0.15)
    parser.add_argument("--eps", type=float, default=0.4)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--S", type=int, default=8)

    parser.add_argument("--stochastic-sampling", action="store_true")
    parser.add_argument("--use-step-wrapper-stochastic", action="store_true")
    parser.add_argument("--gamma-target", type=float, default=None)
    return parser.parse_args()


def _resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    preset = MODEL_PRESETS[args.model_key]
    repo_root = _repo_root()
    if not args.model_name:
        args.model_name = preset.model_name
    if args.num_inference_steps is None:
        args.num_inference_steps = preset.default_steps
    if args.guidance_scale is None:
        args.guidance_scale = preset.guidance_scale
    if not args.prompts_path:
        args.prompts_path = str(
            repo_root / "Fk-Diffusion-Steering" / "text_to_image" / "prompt_files" / "geneval_metadata.jsonl"
        )
    if not args.output_root:
        args.output_root = str(repo_root / "results" / "diffusion_tts_v2")
    if args.model_key == "sd35":
        if not args.stochastic_sampling:
            args.stochastic_sampling = True
        if not args.use_step_wrapper_stochastic:
            args.use_step_wrapper_stochastic = True
        if args.gamma_target is None:
            args.gamma_target = 0.005
    return args


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
    csv_path = exp_root / f"dsearch_winner_geneval_seed{seed}_{shard_tag}.csv"
    jsonl_path = exp_root / f"dsearch_winner_geneval_seed{seed}_{shard_tag}.jsonl"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _metric_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "result": [float(v) for v in arr.tolist()],
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
    }


def _decode_non_sd35_x0(pipeline, x0_preds: torch.Tensor) -> torch.Tensor:
    decode_fn = latent_to_decode_sdxl if isinstance(pipeline, FKDStableDiffusionXL) else latent_to_decode_sd
    with torch.inference_mode():
        decoded = decode_fn(model=pipeline, output_type="pil", latents=x0_preds)
    return decoded.detach()


def install_sd35_step_wrapper_stochastic(scheduler, gamma_target: float) -> None:
    original_step = scheduler.step
    gamma_cap = math.sqrt(2.0) - 1.0
    gamma_eff = float(np.clip(float(gamma_target), 0.0, gamma_cap))

    def wrapped_step(*step_args, **step_kwargs):
        out = original_step(*step_args, **step_kwargs)
        if gamma_eff <= 0.0:
            return out
        timestep = step_args[1] if len(step_args) >= 2 else step_kwargs.get("timestep", None)
        if timestep is None:
            return out
        if hasattr(scheduler, "index_for_timestep"):
            sigma_idx = scheduler.index_for_timestep(timestep)
        else:
            ts = scheduler.timesteps
            sigma_idx = int((ts == timestep).nonzero(as_tuple=True)[0][0].item())
        sigma = scheduler.sigmas[sigma_idx]
        sigma_f = float(sigma.item()) if isinstance(sigma, torch.Tensor) else float(sigma)
        noise_std = sigma_f * math.sqrt(max((1.0 + gamma_eff) ** 2 - 1.0, 0.0))
        if noise_std <= 0.0:
            return out
        prev_sample = out[0] if isinstance(out, tuple) else out.prev_sample
        perturbed = prev_sample + noise_std * torch.randn_like(prev_sample)
        if isinstance(out, tuple):
            return (perturbed,) + out[1:]
        out.prev_sample = perturbed
        return out

    scheduler.step = wrapped_step


def _sd35_step_stateless(scheduler, noise_pred: torch.Tensor, timestep, latents: torch.Tensor) -> torch.Tensor:
    prev_step_index = getattr(scheduler, "_step_index", None)
    prev_begin_index = getattr(scheduler, "_begin_index", None)
    try:
        with torch.inference_mode():
            return scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
    finally:
        if hasattr(scheduler, "_step_index"):
            scheduler._step_index = prev_step_index
        if hasattr(scheduler, "_begin_index"):
            scheduler._begin_index = prev_begin_index


def _predict_noise_non_sd35(pipeline, latents: torch.Tensor, t, prompt_embeds: torch.Tensor, add_text_embeds=None, add_time_ids=None) -> torch.Tensor:
    latent_model_input = torch.cat([latents] * 2) if pipeline.do_classifier_free_guidance else latents
    latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

    added_cond_kwargs = None
    timestep_cond = None
    if isinstance(pipeline, FKDStableDiffusionXL):
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        if pipeline.unet.config.time_cond_proj_dim is not None:
            batch_size = prompt_embeds.shape[0] // 2
            guidance_scale_tensor = torch.tensor(pipeline.guidance_scale - 1).repeat(batch_size)
            timestep_cond = pipeline.get_guidance_scale_embedding(
                guidance_scale_tensor,
                embedding_dim=pipeline.unet.config.time_cond_proj_dim,
            ).to(device=latents.device, dtype=latents.dtype)
    with torch.inference_mode():
        noise_pred = pipeline.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=getattr(pipeline, "cross_attention_kwargs", None),
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
    if pipeline.do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        return (noise_pred_uncond + pipeline.guidance_scale * (noise_pred_text - noise_pred_uncond)).detach()
    return noise_pred.detach()


def _predict_noise_sd35(pipeline, latents: torch.Tensor, t, prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor) -> torch.Tensor:
    latent_model_input = torch.cat([latents] * 2) if pipeline.do_classifier_free_guidance else latents
    timestep = t.expand(latent_model_input.shape[0])
    with torch.inference_mode():
        noise_pred = pipeline.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=getattr(pipeline, "joint_attention_kwargs", None),
            return_dict=False,
        )[0]
    if pipeline.do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        return (noise_pred_uncond + pipeline.guidance_scale * (noise_pred_text - noise_pred_uncond)).detach()
    return noise_pred.detach()


def _prepare_single_prompt_state(args: argparse.Namespace, pipeline, prompt_text: str, sample_seed: int, is_sd35: bool):
    prompt_list = [prompt_text]
    generator = torch.Generator(device=args.device).manual_seed(sample_seed)

    if is_sd35:
        pipeline._guidance_scale = float(args.guidance_scale)
        pipeline._clip_skip = None
        pipeline._joint_attention_kwargs = None
        pipeline._interrupt = False
        with torch.inference_mode():
            pe, ne, ppe, npe = pipeline.encode_prompt(
                prompt=prompt_list,
                prompt_2=None,
                prompt_3=None,
                negative_prompt=None,
                negative_prompt_2=None,
                negative_prompt_3=None,
                do_classifier_free_guidance=True,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                device=pipeline.device,
                clip_skip=None,
                num_images_per_prompt=1,
                max_sequence_length=256,
                lora_scale=None,
            )
        prompt_embeds = torch.cat([ne, pe], dim=0)
        pooled_prompt_embeds = torch.cat([npe, ppe], dim=0)
        with torch.inference_mode():
            latents = pipeline.prepare_latents(
                batch_size=1,
                num_channels_latents=pipeline.transformer.config.in_channels,
                height=pipeline.transformer.config.sample_size * pipeline.vae_scale_factor,
                width=pipeline.transformer.config.sample_size * pipeline.vae_scale_factor,
                dtype=pipeline.transformer.dtype,
                device=pipeline.device,
                generator=[generator],
                latents=None,
            )
        return {
            "prompt_list": prompt_list,
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "add_text_embeds": None,
            "add_time_ids": None,
        }

    pipeline._guidance_scale = float(args.guidance_scale)
    add_text_embeds = None
    add_time_ids = None
    if isinstance(pipeline, FKDStableDiffusionXL):
        with torch.inference_mode():
            pe, ne, ppe, npe = pipeline.encode_prompt(
                prompt=prompt_list,
                prompt_2=None,
                device=pipeline.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=None,
                negative_prompt_2=None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=None,
                clip_skip=None,
            )
        prompt_embeds = torch.cat([ne, pe], dim=0).to(pipeline.device)
        add_text_embeds = torch.cat([npe, ppe], dim=0).to(pipeline.device)
        text_encoder_projection_dim = int(ppe.shape[-1]) if pipeline.text_encoder_2 is None else pipeline.text_encoder_2.config.projection_dim
        h = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        w = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        add_t = pipeline._get_add_time_ids(
            (h, w), (0, 0), (h, w), dtype=prompt_embeds.dtype, text_encoder_projection_dim=text_encoder_projection_dim
        )
        add_time_ids = torch.cat([add_t, add_t], dim=0).to(pipeline.device)
    else:
        with torch.inference_mode():
            prompt_embeds = pipeline._encode_prompt(
                prompt_list,
                pipeline.device,
                1,
                True,
                None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
            )

    with torch.inference_mode():
        latents = pipeline.prepare_latents(
            batch_size=1,
            num_channels_latents=pipeline.unet.config.in_channels,
            height=pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
            width=pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
            dtype=pipeline.unet.dtype,
            device=pipeline.device,
            generator=[generator],
            latents=None,
        )
    return {
        "prompt_list": prompt_list,
        "latents": latents,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": None,
        "add_text_embeds": add_text_embeds,
        "add_time_ids": add_time_ids,
    }


def _score_candidate_x0(args: argparse.Namespace, pipeline, scorer, prompt_text: str, x0_preds: torch.Tensor, is_sd35: bool) -> float:
    if is_sd35:
        image_tensor = decode_latents_to_tensor_sd35(pipeline=pipeline, latents=x0_preds).detach()
    else:
        image_tensor = _decode_non_sd35_x0(pipeline, x0_preds)
    images = pipeline.image_processor.postprocess(image_tensor, output_type="pil")
    score = scorer(images=images, prompts=[prompt_text], timesteps=None)
    if torch.is_tensor(score):
        return float(score.detach().cpu().reshape(-1)[0].item())
    return float(score)


def _run_eps_loop(args: argparse.Namespace, pipeline, scorer, prompt_text: str, state: dict, is_sd35: bool):
    latents = state["latents"]
    prompt_embeds = state["prompt_embeds"]
    pooled_prompt_embeds = state["pooled_prompt_embeds"]
    add_text_embeds = state["add_text_embeds"]
    add_time_ids = state["add_time_ids"]

    timesteps = pipeline.scheduler.timesteps
    extra_step_kwargs = {}
    if not is_sd35 and "eta" in set(inspect.signature(pipeline.scheduler.step).parameters.keys()):
        extra_step_kwargs["eta"] = args.eta

    for i, t in enumerate(timesteps):
        if is_sd35:
            old_noise_pred = _predict_noise_sd35(pipeline, latents, t, prompt_embeds, pooled_prompt_embeds)
        else:
            old_noise_pred = _predict_noise_non_sd35(pipeline, latents, t, prompt_embeds, add_text_embeds, add_time_ids)

        if i >= len(timesteps) - 1:
            if is_sd35:
                latents = _sd35_step_stateless(pipeline.scheduler, old_noise_pred, t, latents)
            else:
                latents = pipeline.scheduler.step(old_noise_pred, t, latents, return_dict=False, **extra_step_kwargs)[0]
            continue

        t_next = timesteps[i + 1]
        pivot = torch.randn_like(latents)

        if args.method in {"eps_greedy", "zero_order"}:
            for _ in range(args.K):
                best_score = -float("inf")
                best_noise = pivot
                for _ in range(args.N):
                    if args.method == "eps_greedy" and torch.rand(1).item() < args.eps:
                        noise_candidate = torch.randn_like(latents)
                    else:
                        to_add = torch.randn_like(latents)
                        to_add = to_add / torch.norm(to_add)
                        scale = torch.rand(1).item() * args.lambda_ * math.sqrt(
                            latents.shape[-1] * latents.shape[-2] * latents.shape[-3]
                        )
                        noise_candidate = pivot + to_add * scale

                    if is_sd35:
                        latents_cand = _sd35_step_stateless(pipeline.scheduler, old_noise_pred, t, latents)
                        x0_preds = compute_x0_preds_sd35(
                            pipeline=pipeline,
                            prev_latents=latents_cand,
                            t=t_next,
                            callback_kwargs={
                                "latents": latents_cand,
                                "prompt_embeds": prompt_embeds,
                                "pooled_prompt_embeds": pooled_prompt_embeds,
                            },
                        )
                    else:
                        latents_cand = pipeline.scheduler.step(
                            old_noise_pred,
                            t,
                            latents,
                            variance_noise=noise_candidate,
                            return_dict=False,
                            **extra_step_kwargs,
                        )[0]
                        cb_kwargs = {"prompt_embeds": prompt_embeds}
                        if add_text_embeds is not None:
                            cb_kwargs["add_text_embeds"] = add_text_embeds
                        if add_time_ids is not None:
                            cb_kwargs["add_time_ids"] = add_time_ids
                        x0_preds = compute_x0_preds_non_sd35(
                            pipeline=pipeline,
                            scheduler=pipeline.scheduler,
                            prev_latents=latents_cand,
                            t=t_next,
                            callback_kwargs=cb_kwargs,
                            eta=args.eta,
                        )

                    cand_score = _score_candidate_x0(args, pipeline, scorer, prompt_text, x0_preds, is_sd35)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_noise = noise_candidate
                pivot = best_noise

        if is_sd35:
            latents = _sd35_step_stateless(pipeline.scheduler, old_noise_pred, t, latents)
        else:
            latents = pipeline.scheduler.step(
                old_noise_pred,
                t,
                latents,
                variance_noise=pivot,
                return_dict=False,
                **extra_step_kwargs,
            )[0]

    return latents


def main() -> None:
    args = _resolve_defaults(parse_args())
    torch.set_grad_enabled(False)

    final_metrics_to_compute = [m for m in args.metrics_to_compute.split("#") if m]
    if not final_metrics_to_compute:
        final_metrics_to_compute = [args.guidance_reward_fn]
    unsupported = [m for m in final_metrics_to_compute if m != "ImageReward"]
    if unsupported:
        raise ValueError(f"Unsupported metrics: {unsupported}")

    prompt_rows = _read_geneval_rows(Path(args.prompts_path).resolve())
    if args.prompt_start_id < 0 or args.prompt_start_id >= len(prompt_rows):
        raise ValueError("--prompt-start-id out of range")
    remaining = len(prompt_rows) - args.prompt_start_id
    requested = remaining if args.num_prompts is None else args.num_prompts
    selected_count = min(remaining, requested)
    selected_rows = prompt_rows[args.prompt_start_id : args.prompt_start_id + selected_count]
    if not selected_rows:
        raise ValueError("No prompts selected")

    guidance_tag = "ir"
    exp_name = f"epsgreedy_{args.model_key}_t{args.num_inference_steps}_{guidance_tag}_geneval"
    exp_root = Path(args.output_root).resolve() / MODEL_PRESETS[args.model_key].model_tag / exp_name
    exp_root.mkdir(parents=True, exist_ok=True)
    seed_dir = exp_root / f"seed={args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    seed_dir.mkdir(parents=True, exist_ok=False)

    args_to_save = vars(args).copy()
    args_to_save["resolved_num_prompts"] = selected_count
    (seed_dir / "args.json").write_text(json.dumps(args_to_save, indent=2), encoding="utf-8")

    pipeline, is_sd35 = build_pipeline(args.model_name, args.device)
    pipeline.set_progress_bar_config(disable=True)

    if not is_sd35:
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler.set_timesteps(args.num_inference_steps, device=pipeline.device)
    else:
        scheduler_cls = pipeline.scheduler.__class__
        init_params = set(inspect.signature(scheduler_cls.__init__).parameters.keys())
        if "stochastic_sampling" in init_params:
            pipeline.scheduler = scheduler_cls.from_config(
                pipeline.scheduler.config,
                stochastic_sampling=(args.stochastic_sampling and not args.use_step_wrapper_stochastic),
            )
        else:
            pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config)
        if args.use_step_wrapper_stochastic:
            install_sd35_step_wrapper_stochastic(pipeline.scheduler, float(args.gamma_target))
        pipeline.scheduler.set_timesteps(args.num_inference_steps, device=pipeline.device)

    ImageRewardScorer = _import_sd_image_reward_scorer()
    scorers = {"ImageReward": ImageRewardScorer(dtype=torch.float32)}

    agg_metrics: Dict[str, Dict[str, float]] = {
        m: {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0} for m in final_metrics_to_compute
    }
    winner_rows: List[dict] = []
    audit_rows: List[dict] = []
    total_prompt_time = 0.0
    total_inference_time = 0.0
    n_prompts = 0

    for local_idx, row in enumerate(tqdm(selected_rows, desc="Prompts", unit="prompt")):
        prompt_id = args.prompt_start_id + local_idx
        prompt_text = str(row.get("prompt", "")).strip()
        if not prompt_text:
            continue

        prompt_dir = seed_dir / f"{prompt_id:05d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "metadata.jsonl").write_text(json.dumps(row), encoding="utf-8")

        prompt_start = time.perf_counter()
        prompt_inference_time = 0.0
        generated_images = []

        for sample_idx in range(args.num_images):
            sample_seed = args.seed + prompt_id * args.num_images + sample_idx
            state = _prepare_single_prompt_state(args, pipeline, prompt_text, sample_seed, is_sd35)
            infer_start = time.perf_counter()
            final_latents = _run_eps_loop(args, pipeline, scorers[args.guidance_reward_fn], prompt_text, state, is_sd35)
            prompt_inference_time += time.perf_counter() - infer_start

            if is_sd35:
                final_tensor = decode_latents_to_tensor_sd35(pipeline=pipeline, latents=final_latents).detach()
            else:
                final_tensor = _decode_non_sd35_x0(pipeline, final_latents)
            final_images = pipeline.image_processor.postprocess(final_tensor, output_type="pil")
            generated_images.append(final_images[0])

        metric_values: Dict[str, List[float]] = {}
        for metric in final_metrics_to_compute:
            metric_tensor = scorers[metric](
                images=generated_images,
                prompts=[prompt_text] * len(generated_images),
                timesteps=None,
            )
            metric_values[metric] = [float(v) for v in metric_tensor.detach().cpu().tolist()]

        prompt_elapsed = time.perf_counter() - prompt_start
        total_prompt_time += prompt_elapsed
        total_inference_time += prompt_inference_time
        n_prompts += 1

        guidance_scores = np.asarray(metric_values[args.guidance_reward_fn], dtype=np.float64)
        order = np.argsort(guidance_scores)[::-1]
        top = order[: min(args.bs, len(order))]
        sorted_images = [generated_images[int(i)] for i in top]

        final_metric_res: Dict[str, dict] = {}
        for metric in final_metrics_to_compute:
            vals = np.asarray(metric_values[metric], dtype=np.float64)[top]
            stats = _metric_stats(vals.tolist())
            final_metric_res[metric] = stats
            agg_metrics[metric]["mean"] += stats["mean"]
            agg_metrics[metric]["max"] += stats["max"]
            agg_metrics[metric]["min"] += stats["min"]
            agg_metrics[metric]["std"] += stats["std"]

        final_metric_res["time_taken"] = float(prompt_elapsed)
        final_metric_res["search_events"] = int(args.K * max(0, args.num_inference_steps - 1))
        final_metric_res["particle_backbone_calls"] = 0
        final_metric_res["kl_proxy_mean"] = 0.0
        final_metric_res["profile_pipeline_wall_time"] = float(prompt_inference_time)
        final_metric_res["profile_reward_eval_time"] = float(max(0.0, prompt_elapsed - prompt_inference_time))
        final_metric_res["profile_diffusion_approx_time"] = float(prompt_inference_time)
        final_metric_res["profile_reward_eval_calls"] = int(len(final_metrics_to_compute))
        final_metric_res["prompt"] = [prompt_text] * len(top)
        final_metric_res["prompt_index"] = int(prompt_id)

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
                "sample_relpath": str(Path(seed_dir.name) / f"{prompt_id:05d}" / "best_of_n_samples" / "00000.png"),
                "image_reward": float(final_metric_res["ImageReward"]["result"][0]),
            }
        )
        audit_rows.append(
            {
                "prompt_id": int(prompt_id),
                "prompt": prompt_text,
                "prompt_wall_time": float(prompt_elapsed),
                "pipeline_wall_time": float(prompt_inference_time),
                "reward_eval_time": float(max(0.0, prompt_elapsed - prompt_inference_time)),
            }
        )

    if n_prompts == 0:
        raise RuntimeError("No prompts were processed.")

    for metric in final_metrics_to_compute:
        for key in ("mean", "max", "min", "std"):
            agg_metrics[metric][key] /= float(n_prompts)

    agg_metrics["compute_audit"] = {
        "total_prompt_wall_time": float(total_prompt_time),
        "total_pipeline_wall_time": float(total_inference_time),
        "avg_prompt_wall_time": float(total_prompt_time / n_prompts),
        "avg_pipeline_wall_time": float(total_inference_time / n_prompts),
        "num_prompts": int(n_prompts),
        "num_inference_steps": int(args.num_inference_steps),
        "model_key": args.model_key,
        "N": int(args.N),
        "K": int(args.K),
    }
    (seed_dir / "final_metrics.json").write_text(json.dumps(agg_metrics), encoding="utf-8")

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
