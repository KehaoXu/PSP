#!/usr/bin/env python3
import argparse
import csv
import inspect
import json
import os
from typing import Dict, List, Sequence

import torch
from tqdm import tqdm
from diffusers import DDIMScheduler, UNet2DConditionModel

from fks_utils import do_eval, get_model
from fkd_diffusers.fkd_pipeline_sd import (
    FKDStableDiffusion,
    latent_to_decode as latent_to_decode_sd,
)
from fkd_diffusers.fkd_pipeline_sd3 import FKDStableDiffusion3
from fkd_diffusers.fkd_pipeline_sdxl import (
    FKDStableDiffusionXL,
    latent_to_decode as latent_to_decode_sdxl,
)


def _append_prompt(prompts: List[str], value) -> None:
    if not isinstance(value, str):
        return
    prompt = value.strip()
    if prompt:
        prompts.append(prompt)


def read_prompts(path: str) -> List[str]:
    prompts: List[str] = []
    ext = os.path.splitext(path)[1].lower()

    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    _append_prompt(prompts, obj.get("prompt"))
        return prompts

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    _append_prompt(prompts, item.get("prompt"))
                elif isinstance(item, str):
                    _append_prompt(prompts, item)
            return prompts

        if isinstance(data, dict):
            if "prompt" in data:
                _append_prompt(prompts, data.get("prompt"))
            else:
                for value in data.values():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                _append_prompt(prompts, item.get("prompt"))
                            elif isinstance(item, str):
                                _append_prompt(prompts, item)
                    elif isinstance(value, dict):
                        _append_prompt(prompts, value.get("prompt"))
                    elif isinstance(value, str):
                        _append_prompt(prompts, value)
            return prompts

        return prompts

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            _append_prompt(prompts, line)
    return prompts


def chunked(items: Sequence[int], chunk_size: int) -> List[List[int]]:
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]


def compute_x0_preds_non_sd35(
    *,
    pipeline,
    scheduler,
    prev_latents: torch.Tensor,
    t,
    callback_kwargs,
    eta: float,
) -> torch.Tensor:
    prompt_embeds = callback_kwargs["prompt_embeds"]
    latent_model_input = (
        torch.cat([prev_latents] * 2)
        if pipeline.do_classifier_free_guidance
        else prev_latents
    )
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    added_cond_kwargs = None
    timestep_cond = None
    if isinstance(pipeline, FKDStableDiffusionXL):
        # SDXL UNet expects extra conditioning fields in added_cond_kwargs.
        added_cond_kwargs = {
            "text_embeds": callback_kwargs.get("add_text_embeds"),
            "time_ids": callback_kwargs.get("add_time_ids"),
        }
        if pipeline.unet.config.time_cond_proj_dim is not None:
            batch_size = prompt_embeds.shape[0] // (
                2 if pipeline.do_classifier_free_guidance else 1
            )
            guidance_scale_tensor = torch.tensor(pipeline.guidance_scale - 1).repeat(
                batch_size
            )
            timestep_cond = pipeline.get_guidance_scale_embedding(
                guidance_scale_tensor,
                embedding_dim=pipeline.unet.config.time_cond_proj_dim,
            ).to(device=prev_latents.device, dtype=prev_latents.dtype)

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
        noise_pred = noise_pred_uncond + pipeline.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

    extra_step_kwargs = {}
    if "eta" in set(inspect.signature(scheduler.step).parameters.keys()):
        extra_step_kwargs["eta"] = eta

    with torch.inference_mode():
        step_dict = scheduler.step(
            noise_pred, t, prev_latents, return_dict=True, **extra_step_kwargs
        )
    return step_dict.pred_original_sample.detach()


def decode_latents_to_tensor_sd35(*, pipeline, latents: torch.Tensor) -> torch.Tensor:
    vae = pipeline.vae
    needs_upcast = vae.dtype == torch.float16 and getattr(vae.config, "force_upcast", False)
    if needs_upcast:
        if hasattr(pipeline, "upcast_vae"):
            pipeline.upcast_vae()
        else:
            vae.to(dtype=torch.float32)

    with torch.inference_mode():
        if getattr(vae, "post_quant_conv", None) is not None:
            decode_param = next(vae.post_quant_conv.parameters())
        else:
            decode_param = next(vae.parameters())
        latents = (latents / vae.config.scaling_factor).to(
            device=decode_param.device, dtype=decode_param.dtype
        )
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
            latents = latents + vae.config.shift_factor
        image = vae.decode(latents, return_dict=False)[0]

    if needs_upcast:
        vae.to(dtype=torch.float16)

    return image.detach()


def compute_x0_preds_sd35(*, pipeline, prev_latents: torch.Tensor, t: torch.Tensor, callback_kwargs):
    prompt_embeds = callback_kwargs["prompt_embeds"]
    pooled_prompt_embeds = callback_kwargs["pooled_prompt_embeds"]
    latent_model_input = (
        torch.cat([prev_latents] * 2) if pipeline.do_classifier_free_guidance else prev_latents
    )
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
        noise_pred = noise_pred_uncond + pipeline.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

    if hasattr(pipeline.scheduler, "index_for_timestep"):
        sigma_idx = pipeline.scheduler.index_for_timestep(t)
    else:
        timesteps = pipeline.scheduler.timesteps
        sigma_idx = int((timesteps == t).nonzero(as_tuple=True)[0][0].item())
    sigma = pipeline.scheduler.sigmas[sigma_idx].to(
        device=prev_latents.device, dtype=prev_latents.dtype
    )
    while sigma.ndim < prev_latents.ndim:
        sigma = sigma.view(*sigma.shape, 1)
    return (prev_latents - sigma * noise_pred).detach()


def build_pipeline(model_name: str, device: str):
    normalized = model_name.strip().lower()
    is_sdxl_dpo = "mhdang/dpo-sdxl-text2image-v1" in normalized
    is_sd15_dpo = "mhdang/dpo-sd1.5-text2image-v1" in normalized
    is_sd35 = normalized in {
        "stable-diffusion-3.5-large",
        "stable-diffusion-v3-5",
        "stable-diffusion-3.5",
        "sd3.5",
    }
    if is_sd35:
        pipeline = FKDStableDiffusion3.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
        ).to(device)
        return pipeline, True

    if is_sdxl_dpo:
        pipeline = FKDStableDiffusionXL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        unet = UNet2DConditionModel.from_pretrained(
            "mhdang/dpo-sdxl-text2image-v1",
            subfolder="unet",
            torch_dtype=torch.float16,
        )
        pipeline.unet = unet
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(device)
        return pipeline, False

    if is_sd15_dpo:
        pipeline = FKDStableDiffusion.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        unet = UNet2DConditionModel.from_pretrained(
            "mhdang/dpo-sd1.5-text2image-v1",
            subfolder="unet",
            torch_dtype=torch.float16,
        )
        pipeline.unet = unet
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(device)
        return pipeline, False

    pipeline = get_model(model_name).to(device)
    return pipeline, False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect per-step ImageReward/HPS(pred_x0) trajectories for prompts."
    )
    parser.add_argument("--prompts-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="stable-diffusion-v1-5")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-start-id", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=64)
    parser.add_argument("--time-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument(
        "--no_hps",
        action="store_true",
        help="Skip HumanPreference metric collection and only store ImageReward.",
    )
    args = parser.parse_args()

    if args.prompt_start_id < 0:
        raise ValueError("--prompt-start-id must be >= 0")
    if args.num_prompts is not None and args.num_prompts < 1:
        raise ValueError("--num-prompts must be >= 1 when provided")
    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be >= 1")
    if args.time_steps < 1:
        raise ValueError("--time-steps must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    all_prompts = read_prompts(args.prompts_path)
    if not all_prompts:
        raise ValueError("No prompts found in the provided prompts file.")
    if args.prompt_start_id >= len(all_prompts):
        raise ValueError(
            f"--prompt-start-id ({args.prompt_start_id}) out of range for dataset size {len(all_prompts)}"
        )

    remaining = len(all_prompts) - args.prompt_start_id
    selected_count = args.num_prompts if args.num_prompts is not None else remaining
    selected_count = min(selected_count, remaining)
    prompt_end_id = args.prompt_start_id + selected_count
    selected_prompts = all_prompts[args.prompt_start_id:prompt_end_id]

    print(f"collecting for prompt ids {args.prompt_start_id} <= id < {prompt_end_id}")

    os.makedirs(args.output_dir, exist_ok=False)
    metrics_csv_path = os.path.join(args.output_dir, "metrics.csv")
    samples_root = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_root, exist_ok=True)

    args_to_save = vars(args).copy()
    args_to_save["prompt_end_id"] = prompt_end_id
    args_to_save["resolved_num_prompts"] = selected_count
    print("Args:", json.dumps(args_to_save, indent=2))
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(args_to_save, f, indent=2)

    pipeline, is_sd35 = build_pipeline(args.model_name, args.device)
    pipeline.set_progress_bar_config(disable=True)
    normalized_model_name = args.model_name.strip().lower()
    is_dpo_model = "mhdang/dpo-" in normalized_model_name
    collect_hps = not args.no_hps
    metrics_to_compute = ["ImageReward"] + (["HumanPreference"] if collect_hps else [])

    total_images = len(selected_prompts) * args.num_seeds
    running_final_ir_sum = 0.0
    running_final_hps_sum = 0.0
    running_final_count = 0
    pct_labels = ["20", "40", "60", "80"]
    pct_fractions = [0.2, 0.4, 0.6, 0.8]
    pct_step_numbers = [
        max(1, min(args.time_steps, int(round(args.time_steps * frac))))
        for frac in pct_fractions
    ]
    pct_step_indices = [step_num - 1 for step_num in pct_step_numbers]
    running_pct_ir_sum = {label: 0.0 for label in pct_labels}
    running_pct_hps_sum = {label: 0.0 for label in pct_labels}
    running_pct_count = {label: 0 for label in pct_labels}

    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as csvfile, tqdm(
        total=total_images, desc="Images", unit="img"
    ) as pbar:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "prompt_id",
                "prompt",
                "seed",
                "step",
                "timestep",
                "image_reward",
                "human_preference",
            ],
        )
        writer.writeheader()

        for local_prompt_idx, prompt in enumerate(selected_prompts):
            prompt_id = args.prompt_start_id + local_prompt_idx
            prompt_sample_dir = os.path.join(samples_root, f"{prompt_id:05d}")
            os.makedirs(prompt_sample_dir, exist_ok=True)

            prompt_seed_base = args.seed + prompt_id * args.num_seeds
            prompt_seeds = [prompt_seed_base + i for i in range(args.num_seeds)]
            for batch in chunked(prompt_seeds, args.batch_size):
                batch_step_bar = tqdm(
                    total=args.time_steps,
                    desc=f"Prompt {prompt_id} batch",
                    unit="step",
                    leave=False,
                )
                prompt_list = [prompt] * len(batch)
                generators = [
                    torch.Generator(device=args.device).manual_seed(seed)
                    for seed in batch
                ]

                step_image_reward: Dict[int, List[float]] = {}
                step_hps: Dict[int, List[float]] = {}
                step_timesteps: Dict[int, int] = {}

                if is_sd35:
                    prev_latents = pipeline.prepare_latents(
                        batch_size=len(batch),
                        num_channels_latents=pipeline.transformer.config.in_channels,
                        height=pipeline.transformer.config.sample_size * pipeline.vae_scale_factor,
                        width=pipeline.transformer.config.sample_size * pipeline.vae_scale_factor,
                        dtype=pipeline.transformer.dtype,
                        device=pipeline.device,
                        generator=list(generators),
                        latents=None,
                    )

                    def capture_callback(_pipeline, step_idx, t, callback_kwargs):
                        nonlocal prev_latents
                        x0_preds = compute_x0_preds_sd35(
                            pipeline=pipeline,
                            prev_latents=prev_latents,
                            t=t,
                            callback_kwargs=callback_kwargs,
                        )
                        image_tensor = decode_latents_to_tensor_sd35(
                            pipeline=pipeline, latents=x0_preds
                        ).detach()
                        images = pipeline.image_processor.postprocess(image_tensor, output_type="pil")
                        metric_res = do_eval(
                            prompt=prompt_list,
                            images=images,
                            metrics_to_compute=metrics_to_compute,
                        )
                        step_image_reward[step_idx] = [
                            float(v) for v in metric_res["ImageReward"]["result"]
                        ]
                        if collect_hps:
                            step_hps[step_idx] = [
                                float(v) for v in metric_res["HumanPreference"]["result"]
                            ]
                        step_timesteps[step_idx] = int(t)
                        prev_latents = callback_kwargs.get("latents", prev_latents)
                        batch_step_bar.update(1)
                        return {}

                    with torch.no_grad():
                        output = pipeline(
                            prompt_list,
                            num_inference_steps=args.time_steps,
                            generator=list(generators),
                            latents=prev_latents,
                            output_type="pil",
                            callback_on_step_end=capture_callback,
                            callback_on_step_end_tensor_inputs=[
                                "latents",
                                "prompt_embeds",
                                "negative_prompt_embeds",
                                "pooled_prompt_embeds",
                                "negative_pooled_prompt_embeds",
                            ],
                        )
                    final_images = output.images if hasattr(output, "images") else output[0]
                else:
                    prev_latents = pipeline.prepare_latents(
                        batch_size=len(batch),
                        num_channels_latents=pipeline.unet.config.in_channels,
                        height=pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                        width=pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                        dtype=pipeline.unet.dtype,
                        device=pipeline.device,
                        generator=generators,
                        latents=None,
                    )
                    # Keep legacy behavior for non-DPO models. For DPO models, use a
                    # dedicated scheduler copy in callback x0 estimation to avoid mutating
                    # the live scheduler state used by the sampling loop.
                    if is_dpo_model:
                        callback_scheduler = DDIMScheduler.from_config(
                            pipeline.scheduler.config
                        )
                        callback_scheduler.set_timesteps(
                            args.time_steps, device=pipeline.device
                        )
                        scoring_scheduler = callback_scheduler
                    else:
                        scoring_scheduler = pipeline.scheduler

                    def capture_callback(_pipeline, step_idx, t, callback_kwargs):
                        nonlocal prev_latents
                        prompt_embeds = callback_kwargs.get("prompt_embeds")
                        if prompt_embeds is None:
                            return {}
                        x0_preds = compute_x0_preds_non_sd35(
                            pipeline=pipeline,
                            scheduler=scoring_scheduler,
                            prev_latents=prev_latents,
                            t=t,
                            callback_kwargs=callback_kwargs,
                            eta=args.eta,
                        )
                        decode_fn = (
                            latent_to_decode_sdxl
                            if isinstance(pipeline, FKDStableDiffusionXL)
                            else latent_to_decode_sd
                        )
                        image_tensor = decode_fn(
                            model=pipeline, output_type="pil", latents=x0_preds
                        ).detach()
                        images = pipeline.image_processor.postprocess(image_tensor, output_type="pil")
                        metric_res = do_eval(
                            prompt=prompt_list,
                            images=images,
                            metrics_to_compute=metrics_to_compute,
                        )
                        step_image_reward[step_idx] = [
                            float(v) for v in metric_res["ImageReward"]["result"]
                        ]
                        if collect_hps:
                            step_hps[step_idx] = [
                                float(v) for v in metric_res["HumanPreference"]["result"]
                            ]
                        step_timesteps[step_idx] = int(t)
                        prev_latents = callback_kwargs.get("latents", prev_latents)
                        batch_step_bar.update(1)
                        return {}

                    with torch.no_grad():
                        callback_inputs = [
                            "latents",
                            "prompt_embeds",
                            "negative_prompt_embeds",
                        ]
                        if isinstance(pipeline, FKDStableDiffusionXL):
                            callback_inputs.extend(
                                [
                                    "add_text_embeds",
                                    "negative_pooled_prompt_embeds",
                                    "add_time_ids",
                                    "negative_add_time_ids",
                                ]
                            )
                        output = pipeline(
                            prompt_list,
                            num_inference_steps=args.time_steps,
                            eta=args.eta,
                            generator=generators,
                            latents=prev_latents,
                            callback_on_step_end=capture_callback,
                            callback_on_step_end_tensor_inputs=callback_inputs,
                        )
                    final_images = output.images if hasattr(output, "images") else output[0]
                batch_step_bar.close()

                for seed, image in zip(batch, final_images):
                    image.save(os.path.join(prompt_sample_dir, f"{seed:05d}.png"))

                pbar.update(len(batch))
                final_step_idx = args.time_steps - 1
                final_ir_scores = step_image_reward.get(final_step_idx)
                final_hps_scores = step_hps.get(final_step_idx) if collect_hps else None
                if final_ir_scores is not None:
                    running_final_ir_sum += float(sum(final_ir_scores))
                    running_final_count += len(final_ir_scores)
                    running_ir_mean = running_final_ir_sum / max(1, running_final_count)
                    postfix = {
                        "final_IR_mean": f"{running_ir_mean:.4f}",
                        "final_n": running_final_count,
                    }
                    if collect_hps and final_hps_scores is not None:
                        running_final_hps_sum += float(sum(final_hps_scores))
                        running_hps_mean = running_final_hps_sum / max(1, running_final_count)
                        postfix["final_HPS_mean"] = f"{running_hps_mean:.4f}"

                    # Update running averages for 20/40/60/80 checkpoints.
                    for label, step_idx_pct in zip(pct_labels, pct_step_indices):
                        ir_scores_pct = step_image_reward.get(step_idx_pct)
                        if ir_scores_pct is None:
                            continue
                        running_pct_ir_sum[label] += float(sum(ir_scores_pct))
                        running_pct_count[label] += len(ir_scores_pct)
                        if collect_hps:
                            hps_scores_pct = step_hps.get(step_idx_pct)
                            if hps_scores_pct is not None:
                                running_pct_hps_sum[label] += float(sum(hps_scores_pct))

                    ir_parts = []
                    hps_parts = []
                    for label in pct_labels:
                        cnt = running_pct_count[label]
                        if cnt <= 0:
                            continue
                        ir_parts.append(f"{label}%:{(running_pct_ir_sum[label] / cnt):.3f}")
                        if collect_hps:
                            hps_parts.append(f"{label}%:{(running_pct_hps_sum[label] / cnt):.3f}")
                    if ir_parts:
                        postfix["IR@20/40/60/80"] = " ".join(ir_parts)
                    if collect_hps and hps_parts:
                        postfix["HPS@20/40/60/80"] = " ".join(hps_parts)
                    pbar.set_postfix(postfix)

                    # Detailed terminal-only sanity block for the most recent batch:
                    # - first 4 seeds values at 20/40/60/80/100
                    # - batch means at 20/40/60/80/100
                    # - running means at 20/40/60/80/100
                    point_labels = pct_labels + ["100"]
                    point_step_numbers = pct_step_numbers + [args.time_steps]
                    point_step_indices = pct_step_indices + [args.time_steps - 1]
                    snapshot_seed_count = min(4, len(batch))
                    snapshot_seeds = batch[:snapshot_seed_count]

                    pbar.write(
                        f"[batch sanity] prompt_id={prompt_id} seeds={snapshot_seeds} "
                        f"steps={point_step_numbers}"
                    )

                    for local_idx, seed in enumerate(snapshot_seeds):
                        ir_chunks = []
                        hps_chunks = []
                        for label, idx_point in zip(point_labels, point_step_indices):
                            ir_vals = step_image_reward.get(idx_point)
                            if ir_vals is not None and local_idx < len(ir_vals):
                                ir_chunks.append(f"{label}%:{float(ir_vals[local_idx]):.4f}")
                            if collect_hps:
                                hps_vals = step_hps.get(idx_point)
                                if hps_vals is not None and local_idx < len(hps_vals):
                                    hps_chunks.append(f"{label}%:{float(hps_vals[local_idx]):.4f}")
                        line = f"  seed={seed} IR[{', '.join(ir_chunks)}]"
                        if collect_hps:
                            line += f" HPS[{', '.join(hps_chunks)}]"
                        pbar.write(line)

                    batch_ir_parts = []
                    batch_hps_parts = []
                    running_ir_parts = []
                    running_hps_parts = []
                    for label, idx_point in zip(point_labels, point_step_indices):
                        ir_vals = step_image_reward.get(idx_point)
                        if ir_vals is not None and len(ir_vals) > 0:
                            batch_ir_parts.append(f"{label}%:{(float(sum(ir_vals)) / len(ir_vals)):.4f}")
                        if collect_hps:
                            hps_vals = step_hps.get(idx_point)
                            if hps_vals is not None and len(hps_vals) > 0:
                                batch_hps_parts.append(f"{label}%:{(float(sum(hps_vals)) / len(hps_vals)):.4f}")

                        if label == "100":
                            if running_final_count > 0:
                                running_ir_parts.append(f"{label}%:{(running_final_ir_sum / running_final_count):.4f}")
                                if collect_hps:
                                    running_hps_parts.append(
                                        f"{label}%:{(running_final_hps_sum / running_final_count):.4f}"
                                    )
                        else:
                            cnt = running_pct_count[label]
                            if cnt > 0:
                                running_ir_parts.append(f"{label}%:{(running_pct_ir_sum[label] / cnt):.4f}")
                                if collect_hps:
                                    running_hps_parts.append(f"{label}%:{(running_pct_hps_sum[label] / cnt):.4f}")

                    pbar.write(f"  batch_mean IR[{', '.join(batch_ir_parts)}]")
                    if collect_hps:
                        pbar.write(f"  batch_mean HPS[{', '.join(batch_hps_parts)}]")
                    pbar.write(f"  running_mean IR[{', '.join(running_ir_parts)}] n={running_final_count}")
                    if collect_hps:
                        pbar.write(f"  running_mean HPS[{', '.join(running_hps_parts)}] n={running_final_count}")

                for step_idx in range(args.time_steps):
                    image_rewards = step_image_reward.get(step_idx)
                    if image_rewards is None:
                        continue
                    hps_scores = (
                        step_hps.get(step_idx)
                        if collect_hps
                        else [None] * len(image_rewards)
                    )
                    if hps_scores is None:
                        continue
                    timestep_value = step_timesteps.get(step_idx, None)
                    for seed, ir_score, hps_score in zip(batch, image_rewards, hps_scores):
                        writer.writerow(
                            {
                                "prompt_id": prompt_id,
                                "prompt": prompt,
                                "seed": seed,
                                "step": step_idx + 1,
                                "timestep": timestep_value,
                                "image_reward": ir_score,
                                "human_preference": hps_score,
                            }
                        )

    print(f"Saved metrics to {metrics_csv_path}")


if __name__ == "__main__":
    main()
