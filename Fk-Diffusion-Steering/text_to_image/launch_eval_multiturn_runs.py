#!/usr/bin/env python3
# primary generation script (multi-turn online search)
import os
import json
import inspect
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, UNet2DConditionModel
from tqdm import tqdm

import sys

sys.path.append("fkd_diffusers")

from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL
from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion
from fkd_diffusers.fkd_pipeline_sd3 import FKDStableDiffusion3
from fkd_diffusers.rewards import do_image_reward, get_reward_function
from fks_utils import do_eval


def load_geneval_metadata(prompt_path, max_prompts=None):
    if prompt_path.endswith(".json"):
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        assert prompt_path.endswith(".jsonl")
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    assert isinstance(data, list)
    prompt_key = "prompt"
    if prompt_key not in data[0]:
        assert "text" in data[0], "Prompt data should have 'prompt' or 'text' key"
        for item in data:
            item["prompt"] = item["text"]
    if max_prompts is not None:
        data = data[:max_prompts]
    return data


def parse_int_list(value: str) -> List[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(item) for item in value.split(",") if item.strip()]


def decode_latents_to_pil(*, pipeline, latents: torch.Tensor) -> List[Image.Image]:
    vae = pipeline.vae
    needs_upcast = vae.dtype == torch.float16 and getattr(vae.config, "force_upcast", False)
    if needs_upcast:
        if hasattr(pipeline, "upcast_vae"):
            pipeline.upcast_vae()
        else:
            vae.to(dtype=torch.float32)

    with torch.no_grad():
        # Match latent dtype/device to VAE decode parameters across SD/SDXL/SD3.5.
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

    return pipeline.image_processor.postprocess(image, output_type="pil")


def build_reward_fn(*, pipeline, reward_name: str, prompt: str):
    def reward_fn(x0_preds: torch.Tensor) -> torch.Tensor:
        rewards: List[float] = []
        for latent in x0_preds:
            images = decode_latents_to_pil(pipeline=pipeline, latents=latent.unsqueeze(0))
            if reward_name == "ImageReward":
                reward = do_image_reward(images=images, prompts=[prompt])[0]
            else:
                reward = get_reward_function(reward_name, images=images, prompts=[prompt])[0]
            rewards.append(float(reward))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.tensor(rewards, device=x0_preds.device, dtype=torch.float32)

    return reward_fn


def apply_indices(
    tensor: Optional[torch.Tensor],
    indices: Optional[torch.Tensor],
    *,
    batch_size: int,
) -> Optional[torch.Tensor]:
    if tensor is None or indices is None:
        return tensor
    if indices.device != tensor.device:
        indices = indices.to(tensor.device)
    # Some callback tensors in SDXL are broadcast constants (e.g., shape[0] == 1)
    # and are not aligned to particle batch size. Leave those untouched.
    first_dim = int(tensor.shape[0]) if tensor.ndim > 0 else 0
    if first_dim == batch_size * 2:
        expanded = torch.cat([indices, indices + batch_size])
        return tensor[expanded]
    if first_dim == batch_size:
        return tensor[indices]
    return tensor


def compute_x0_preds(
    *,
    pipeline,
    prev_latents: torch.Tensor,
    t: torch.Tensor,
    callback_kwargs: Dict[str, torch.Tensor],
    eta: float,
) -> torch.Tensor:
    prompt_embeds = callback_kwargs["prompt_embeds"]
    latent_model_input = (
        torch.cat([prev_latents] * 2) if pipeline.do_classifier_free_guidance else prev_latents
    )

    if isinstance(pipeline, FKDStableDiffusion3):
        timestep = t.expand(latent_model_input.shape[0])
        pooled_prompt_embeds = callback_kwargs["pooled_prompt_embeds"]
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

        # FlowMatch scheduler does not return pred_original_sample.
        # Use a stable x0 proxy from the current sigma.
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
        return prev_latents - sigma * noise_pred

    latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

    added_cond_kwargs = None
    timestep_cond = None
    if isinstance(pipeline, FKDStableDiffusionXL):
        added_cond_kwargs = {
            "text_embeds": callback_kwargs.get("add_text_embeds"),
            "time_ids": callback_kwargs.get("add_time_ids"),
        }
        if pipeline.unet.config.time_cond_proj_dim is not None:
            batch_size = prompt_embeds.shape[0] // (2 if pipeline.do_classifier_free_guidance else 1)
            guidance_scale_tensor = torch.tensor(pipeline.guidance_scale - 1).repeat(batch_size)
            timestep_cond = pipeline.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipeline.unet.config.time_cond_proj_dim
            ).to(device=prev_latents.device, dtype=prev_latents.dtype)

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
    if "eta" in set(inspect.signature(pipeline.scheduler.step).parameters.keys()):
        extra_step_kwargs["eta"] = eta

    step_dict = pipeline.scheduler.step(
        noise_pred, t, prev_latents, return_dict=True, **extra_step_kwargs
    )
    return step_dict.pred_original_sample


def _validate_strategy_args(args) -> None:
    if args.search_mode not in {"scheduled", "dynamic"}:
        raise ValueError("--search-mode must be one of: scheduled, dynamic")
    if args.k_init < 1:
        raise ValueError("--k-init must be >= 1")
    if args.num_inference_steps < 2:
        raise ValueError("--num-inference-steps must be >= 2")

    args.cutoff_times_list = parse_int_list(args.cutoff_times)
    if not args.cutoff_times_list:
        raise ValueError("--cutoff-times cannot be empty")
    if any(t <= 0 or t >= args.num_inference_steps for t in args.cutoff_times_list):
        raise ValueError("all cutoff times must satisfy 0 < t < num_inference_steps")
    if args.cutoff_times_list != sorted(args.cutoff_times_list):
        raise ValueError("--cutoff-times must be strictly increasing")

    if args.search_mode == "scheduled":
        args.remaining_particles_list = parse_int_list(args.remaining_particles)
        if len(args.remaining_particles_list) != len(args.cutoff_times_list):
            raise ValueError("--remaining-particles length must match --cutoff-times")
        last = args.k_init
        for keep in args.remaining_particles_list:
            if keep < 1 or keep >= last:
                raise ValueError(
                    "scheduled mode requires strictly decreasing positive remaining particles"
                )
            last = keep
    else:
        args.eps_list = parse_float_list(args.eps_list)
        if len(args.eps_list) != len(args.cutoff_times_list):
            raise ValueError("--eps-list length must match --cutoff-times")
        if args.best_of_n < 1:
            raise ValueError("--best-of-n must be >= 1 for dynamic mode")

        # Same validity condition used by search_dynamic_eps_strategies_vs_baselines.py
        budget = args.best_of_n * args.num_inference_steps
        first_cutoff = args.cutoff_times_list[0]
        min_required = args.k_init * first_cutoff + (args.num_inference_steps - first_cutoff)
        if min_required > budget:
            raise ValueError(
                "invalid dynamic setup: cannot carry K to first cutoff and at least one to end "
                f"under budget N*T (required={min_required}, budget={budget})"
            )


def _run_multiturn_online(
    *,
    pipeline,
    prompt: str,
    args,
    generators: Sequence[torch.Generator],
):
    k_init = args.k_init
    total_t = args.num_inference_steps
    cutoff_times = args.cutoff_times_list
    cutoff_set = set(cutoff_times)

    reward_fn = build_reward_fn(
        pipeline=pipeline, reward_name=args.guidance_reward_fn, prompt=prompt
    )

    current_seed_ids = list(range(k_init))
    consumed = 0
    budget = args.best_of_n * total_t if args.search_mode == "dynamic" else None

    if args.search_mode == "scheduled":
        cutoff_to_keep = {t: int(k) for t, k in zip(cutoff_times, args.remaining_particles_list)}
    else:
        cutoff_to_eps = {int(t): float(e) for t, e in zip(cutoff_times, args.eps_list)}
        next_cutoff_by_step = {}
        for i, t in enumerate(cutoff_times):
            next_t = cutoff_times[i + 1] if i + 1 < len(cutoff_times) else total_t
            next_cutoff_by_step[int(t)] = int(next_t)

    callback_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    if isinstance(pipeline, FKDStableDiffusion3):
        callback_inputs.extend(
            [
                "pooled_prompt_embeds",
                "negative_pooled_prompt_embeds",
            ]
        )
    elif isinstance(pipeline, FKDStableDiffusionXL):
        callback_inputs.extend(
            [
                "add_text_embeds",
                "negative_pooled_prompt_embeds",
                "add_time_ids",
                "negative_add_time_ids",
            ]
        )

    prev_latents: Optional[torch.Tensor] = None
    run_invalid = False
    last_rewards_at_cutoff: Optional[torch.Tensor] = None

    def callback_on_step_end(_pipeline, step_idx, timestep_value, callback_kwargs):
        nonlocal prev_latents, current_seed_ids, consumed, run_invalid, last_rewards_at_cutoff
        latents = callback_kwargs["latents"]
        step_number = step_idx + 1

        # Dynamic budget accounting is step-based and happens on every step.
        if args.search_mode == "dynamic":
            survivors_now = len(current_seed_ids)
            consumed += survivors_now
            remaining_budget = budget - consumed
        else:
            remaining_budget = None

        # Pruning logic is only applied at cutoffs.
        if step_number not in cutoff_set:
            prev_latents = latents
            return {"latents": latents}

        if prev_latents is None:
            raise RuntimeError("prev_latents is not initialized")

        x0_preds = compute_x0_preds(
            pipeline=pipeline,
            prev_latents=prev_latents,
            t=timestep_value,
            callback_kwargs=callback_kwargs,
            eta=args.eta,
        )
        rewards = reward_fn(x0_preds)
        last_rewards_at_cutoff = rewards

        batch_size_before = latents.shape[0]
        indices: Optional[torch.Tensor] = None

        if args.search_mode == "scheduled":
            keep_k = cutoff_to_keep[step_number]
            keep = max(1, min(int(keep_k), len(current_seed_ids)))
            order = torch.argsort(rewards, descending=True)
            indices = order[:keep]
            latents = latents[indices]
            rewards = rewards[indices]
            current_seed_ids = [current_seed_ids[i] for i in indices.tolist()]
        else:
            # Exact semantics from search_dynamic_eps_strategies_vs_baselines.py:
            # 1) eps threshold keep
            # 2) budget feasibility prune at cutoff only
            order = torch.argsort(rewards, descending=True)
            ordered_vals = rewards[order]
            ordered_ids = [current_seed_ids[i] for i in order.tolist()]

            if len(ordered_ids) > 1:
                eps = cutoff_to_eps[step_number]
                top_val = float(ordered_vals[0].item())
                keep_mask = ordered_vals >= (top_val - eps)
                kept_positions = [
                    idx for idx, keep in enumerate(keep_mask.detach().cpu().tolist()) if keep
                ]
                if len(kept_positions) == 0:
                    kept_positions = [0]
                kept_tensor_positions = torch.tensor(
                    kept_positions, device=order.device, dtype=torch.long
                )
                indices = order[kept_tensor_positions]
                latents = latents[indices]
                rewards = rewards[indices]
                current_seed_ids = [ordered_ids[idx] for idx in kept_positions]

            # Budget feasibility check at cutoff.
            next_cutoff = next_cutoff_by_step[step_number]
            seg_len = next_cutoff - step_number
            tail_len = total_t - next_cutoff
            required_if_keep_all = len(current_seed_ids) * seg_len + tail_len

            if required_if_keep_all > remaining_budget:
                remain_to_end = total_t - step_number
                if remain_to_end <= 0:
                    keep_n = 1
                else:
                    keep_n = int(remaining_budget // remain_to_end)

                if keep_n < 1:
                    run_invalid = True
                    prev_latents = latents
                    return {"latents": latents}

                keep_n = min(keep_n, len(current_seed_ids))
                order2 = torch.argsort(rewards, descending=True)
                indices2 = order2[:keep_n]
                latents = latents[indices2]
                rewards = rewards[indices2]
                current_seed_ids = [current_seed_ids[i] for i in indices2.tolist()]
                indices = indices2 if indices is None else indices[indices2]

        outputs = {"latents": latents}
        if indices is not None:
            for key in (
                "prompt_embeds",
                "negative_prompt_embeds",
                "pooled_prompt_embeds",
                "add_text_embeds",
                "negative_pooled_prompt_embeds",
                "add_time_ids",
                "negative_add_time_ids",
            ):
                if key in callback_kwargs:
                    outputs[key] = apply_indices(
                        callback_kwargs[key], indices, batch_size=batch_size_before
                    )
        prev_latents = outputs["latents"]
        return outputs

    with torch.no_grad():
        if isinstance(pipeline, FKDStableDiffusion3):
            latent_in_channels = pipeline.transformer.config.in_channels
            sample_size = pipeline.transformer.config.sample_size
            latent_dtype = pipeline.transformer.dtype
        else:
            latent_in_channels = pipeline.unet.config.in_channels
            sample_size = pipeline.unet.config.sample_size
            latent_dtype = pipeline.unet.dtype

        init_latents = pipeline.prepare_latents(
            batch_size=k_init,
            num_channels_latents=latent_in_channels,
            height=sample_size * pipeline.vae_scale_factor,
            width=sample_size * pipeline.vae_scale_factor,
            dtype=latent_dtype,
            device=pipeline.device,
            generator=generators,
            latents=None,
        )
        prev_latents = init_latents
        if isinstance(pipeline, FKDStableDiffusion3):
            output = pipeline(
                [prompt] * k_init,
                num_inference_steps=total_t,
                generator=list(generators),
                latents=init_latents,
                output_type="latent",
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_inputs,
            )
        else:
            output = pipeline(
                [prompt] * k_init,
                num_inference_steps=total_t,
                eta=args.eta,
                generator=list(generators),
                latents=init_latents,
                output_type="latent",
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_inputs,
            )

    if run_invalid:
        return []

    final_latents = output[0]
    final_images = decode_latents_to_pil(pipeline=pipeline, latents=final_latents)
    return final_images


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    _validate_strategy_args(args)
    args.num_particles = args.k_init

    prompt_data = load_geneval_metadata(args.prompt_path)

    if "stable-diffusion-3.5" in args.model_name:
        print("Using SD3.5")
        try:
            pipe = FKDStableDiffusion3.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load SD3.5. Ensure your HF token is configured and you accepted model terms at "
                "https://huggingface.co/stabilityai/stable-diffusion-3.5-large"
            ) from exc
    elif "xl" in args.model_name and "dpo" not in args.model_name:
        print("Using SDXL")
        pipe = FKDStableDiffusionXL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        )
    elif "mhdang/dpo" in args.model_name and "xl" in args.model_name:
        pipe = FKDStableDiffusionXL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        unet_id = "mhdang/dpo-sdxl-text2image-v1"
        unet = UNet2DConditionModel.from_pretrained(
            unet_id, subfolder="unet", torch_dtype=torch.float16
        )
        pipe.unet = unet
    elif "mhdang/dpo" in args.model_name and "xl" not in args.model_name:
        pipe = FKDStableDiffusion.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        unet_id = "mhdang/dpo-sd1.5-text2image-v1"
        unet = UNet2DConditionModel.from_pretrained(
            unet_id, subfolder="unet", torch_dtype=torch.float16
        )
        pipe.unet = unet
    else:
        print("Using SD")
        pipe = FKDStableDiffusion.from_pretrained(args.model_name, torch_dtype=torch.float16)

    if not isinstance(pipe, FKDStableDiffusion3):
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    if args.output_name:
        # Treat output_name as a parent directory and create a per-seed run folder inside it.
        os.makedirs(args.output_name, exist_ok=True)
        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(args.output_name, f"seed={args.seed}_{cur_time}")
        os.makedirs(output_dir, exist_ok=False)
    else:
        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(args.output_dir, cur_time)
        os.makedirs(output_dir, exist_ok=False)

    arg_path = os.path.join(output_dir, "args.json")
    with open(arg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    metrics_to_compute = args.metrics_to_compute.split("#")
    do_eval(
        prompt=["test"],
        images=[Image.new("RGB", (224, 224))],
        metrics_to_compute=metrics_to_compute,
    )

    metrics_arr = {metric: dict(mean=0, max=0, min=0, std=0) for metric in metrics_to_compute}
    n_samples = 0
    average_time = 0.0

    for prompt_idx, item in enumerate(tqdm(prompt_data)):
        prompt_text = item["prompt"]
        start_time = datetime.now()

        prompt_path = os.path.join(output_dir, f"{prompt_idx:0>5}")
        os.makedirs(prompt_path, exist_ok=True)

        with open(os.path.join(prompt_path, "metadata.jsonl"), "w", encoding="utf-8") as f:
            json.dump(item, f)

        base_seed = args.seed + prompt_idx * args.k_init
        generators = [
            torch.Generator(device=device).manual_seed(base_seed + idx)
            for idx in range(args.k_init)
        ]

        images = _run_multiturn_online(
            pipeline=pipe,
            prompt=prompt_text,
            args=args,
            generators=generators,
        )
        if len(images) == 0:
            # Maintain output contract even for invalid dynamic case.
            images = [Image.new("RGB", (512, 512))]

        prompt_list = [prompt_text] * len(images)
        results = do_eval(prompt=prompt_list, images=images, metrics_to_compute=metrics_to_compute)
        end_time = datetime.now()
        time_taken = end_time - start_time

        results["time_taken"] = time_taken.total_seconds()
        results["prompt"] = prompt_list
        results["prompt_index"] = prompt_idx

        n_samples += 1
        average_time += time_taken.total_seconds()
        print(f"Time taken: {average_time / n_samples}")

        guidance_reward = np.array(results[args.guidance_reward_fn]["result"])
        sorted_idx = np.argsort(guidance_reward)[::-1]
        images = [images[i] for i in sorted_idx]
        for metric in metrics_to_compute:
            results[metric]["result"] = [results[metric]["result"][i] for i in sorted_idx]

        for metric in metrics_to_compute:
            metrics_arr[metric]["mean"] += results[metric]["mean"]
            metrics_arr[metric]["max"] += results[metric]["max"]
            metrics_arr[metric]["min"] += results[metric]["min"]
            metrics_arr[metric]["std"] += results[metric]["std"]

        for metric in metrics_to_compute:
            print(metric, metrics_arr[metric]["mean"] / n_samples, metrics_arr[metric]["max"] / n_samples)

        if args.save_individual_images:
            sample_path = os.path.join(prompt_path, "samples")
            os.makedirs(sample_path, exist_ok=True)
            for image_idx, image in enumerate(images):
                image.save(os.path.join(sample_path, f"{image_idx:05}.png"))

            best_of_n_sample_path = os.path.join(prompt_path, "best_of_n_samples")
            os.makedirs(best_of_n_sample_path, exist_ok=True)
            for image_idx, image in enumerate(images[:1]):
                image.save(os.path.join(best_of_n_sample_path, f"{image_idx:05}.png"))

        with open(os.path.join(prompt_path, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f)

        _, ax = plt.subplots(1, len(images), figsize=(max(1, len(images)) * 5, 5))
        axes = np.atleast_1d(ax)
        for i, image in enumerate(images):
            axes[i].imshow(image)
            axes[i].axis("off")
        plt.suptitle(prompt_text)
        image_fpath = os.path.join(prompt_path, "grid.png")
        plt.savefig(image_fpath)
        plt.close()

    for metric in metrics_to_compute:
        metrics_arr[metric]["mean"] /= n_samples
        metrics_arr[metric]["max"] /= n_samples
        metrics_arr[metric]["min"] /= n_samples
        metrics_arr[metric]["std"] /= n_samples

    with open(os.path.join(output_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_arr, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="geneval_outputs")
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="If provided, overrides output_dir root folder name/path.",
    )
    parser.add_argument("--save_individual_images", type=bool, default=True)
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--use_smc", action="store_true")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward")
    parser.add_argument(
        "--metrics_to_compute",
        type=str,
        default="ImageReward#HumanPreference",
        help="# separated list of metrics",
    )
    parser.add_argument("--prompt_path", type=str, default="geneval_metadata.jsonl")
    parser.add_argument("--model_idx", type=int, default=0, help="Used for selecting model and configuration")
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--lmbda", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adaptive_resampling", action="store_true")
    parser.add_argument("--resample_frequency", type=int, default=5)
    parser.add_argument("--resample_t_start", type=int, default=5)
    parser.add_argument("--resample_t_end", type=int, default=30)
    parser.add_argument("--potential_type", type=str, default="diff")

    # Multi-turn-specific args.
    parser.add_argument("--search-mode", type=str, default="scheduled", choices=["scheduled", "dynamic"])
    parser.add_argument("--k-init", type=int, default=32)
    parser.add_argument("--cutoff-times", type=str, default="4,8,16,32")
    parser.add_argument("--remaining-particles", type=str, default="8,4,2,1")
    parser.add_argument("--eps-list", type=str, default="0.25,0.25,0.1,0.05")
    parser.add_argument("--best-of-n", type=int, default=4)

    args = parser.parse_args()
    print(args.adaptive_resampling)

    if args.prompt_path == "geneval_metadata.jsonl":
        args.save_individual_images = True

    if args.model_idx % 4 == 0:
        args.num_particles = 2
    elif args.model_idx % 4 == 1:
        args.num_particles = 3
    elif args.model_idx % 4 == 2:
        args.num_particles = 4
    elif args.model_idx % 4 == 3:
        args.num_particles = 8
    else:
        raise ValueError("Unknown model index")

    if args.model_idx in [0, 1, 2, 3]:
        args.model_name = "stabilityai/stable-diffusion-2-1"
    elif args.model_idx in [4, 5, 6, 7]:
        args.model_name = "runwayml/stable-diffusion-v1-5"
    elif args.model_idx in [8, 9, 10, 11]:
        args.model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    elif args.model_idx in [12, 13, 14, 15]:
        args.model_name = "CompVis/stable-diffusion-v1-4"
    elif args.model_idx in [16, 17, 18, 19]:
        args.model_name = "stabilityai/stable-diffusion-3.5-large"
    elif args.model_idx in [99]:
        args.model_name = "kvablack/ddpo-alignment"
        args.num_particles = 4
    elif args.model_idx == 100:
        args.model_name = "mhdang/dpo-sd1.5-text2image-v1"
        args.num_particles = 4
    elif args.model_idx == 101:
        args.model_name = "mhdang/dpo-sdxl-text2image-v1"
        args.num_particles = 4
    else:
        raise ValueError(f"Unknown model index {args.model_idx}")

    if not args.output_name:
        args.output_dir = args.prompt_path.replace(".json", "_outputs")
    return args


if __name__ == "__main__":
    args = get_args()
    # Match launch_eval_runs.py behavior exactly: always run three seeds.
    for seed in [42, 43, 44]:
        args.seed = seed
        main(args)
