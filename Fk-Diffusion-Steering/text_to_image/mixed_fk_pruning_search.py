#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

import inspect

from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline

from fkd_diffusers.rewards import do_image_reward, get_reward_function


def read_prompts(path: str, limit: int) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            prompt = line.strip()
            if not prompt:
                continue
            prompts.append(prompt)
            if len(prompts) >= limit:
                break
    return prompts


def parse_int_list(value: str) -> List[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def mean_ci(values: Iterable[float]) -> Tuple[float, float, float]:
    values = np.asarray(list(values), dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    ci = 1.96 * (sd / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, sd, ci


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


def load_pipeline(model_name: str, device: str):
    if model_name == "stable-diffusion-xl":
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        )
    elif model_name == "stable-diffusion-v1-5":
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
    elif model_name == "stable-diffusion-v1-4":
        pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
        )
    elif model_name == "stable-diffusion-2-1":
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    return pipeline.to(device)


def decode_latents_to_pil(*, pipeline, latents: torch.Tensor):
    vae = pipeline.vae
    needs_upcast = (
        vae.dtype == torch.float16 and getattr(vae.config, "force_upcast", False)
    )
    if needs_upcast:
        if hasattr(pipeline, "upcast_vae"):
            pipeline.upcast_vae()
        else:
            vae.to(dtype=torch.float32)

    with torch.no_grad():
        latents = (latents / vae.config.scaling_factor).to(dtype=vae.dtype)
        image = vae.decode(latents, return_dict=False)[0]

    if needs_upcast:
        vae.to(dtype=torch.float16)

    return pipeline.image_processor.postprocess(image, output_type="pil")


def fk_resample(
    *,
    latents: torch.Tensor,
    rewards: torch.Tensor,
    state: Dict[str, torch.Tensor],
    lmbda: float,
    potential_type: str,
    adaptive_resampling: bool,
    step_idx: int,
    time_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    population_rs = state["population_rs"]
    product_of_potentials = state["product_of_potentials"]
    num_particles = latents.shape[0]

    if potential_type == "max":
        rs_candidates = torch.max(rewards, population_rs)
        w = torch.exp(lmbda * rs_candidates)
    elif potential_type == "add":
        rs_candidates = rewards + population_rs
        w = torch.exp(lmbda * rs_candidates)
    elif potential_type == "diff":
        rs_candidates = rewards
        diffs = rs_candidates - population_rs
        w = torch.exp(lmbda * diffs)
    elif potential_type == "rt":
        rs_candidates = rewards
        w = torch.exp(lmbda * rs_candidates)
    else:
        raise ValueError(f"Unknown potential_type: {potential_type}")

    if step_idx == time_steps - 1 and potential_type in {"max", "add", "rt"}:
        w = torch.exp(lmbda * rs_candidates) / product_of_potentials

    w = torch.clamp(w, 0, 1e10)
    w[torch.isnan(w)] = 0.0

    resampled = False
    indices = None
    if adaptive_resampling or step_idx == time_steps - 1:
        normalized_w = w / w.sum()
        ess = 1.0 / (normalized_w.pow(2).sum())
        if ess < 0.5 * num_particles:
            indices = torch.multinomial(w, num_samples=num_particles, replacement=True)
            resampled = True
    else:
        indices = torch.multinomial(w, num_samples=num_particles, replacement=True)
        resampled = True

    if resampled and indices is not None:
        latents = latents[indices]
        rs_candidates = rs_candidates[indices]
        product_of_potentials = product_of_potentials[indices] * w[indices]

    state["population_rs"] = rs_candidates
    state["product_of_potentials"] = product_of_potentials

    return latents, rs_candidates, indices


def apply_indices(
    tensor: Optional[torch.Tensor],
    indices: Optional[torch.Tensor],
    *,
    batch_size: int,
) -> Optional[torch.Tensor]:
    if tensor is None or indices is None:
        return tensor
    if tensor.shape[0] == batch_size * 2:
        expanded = torch.cat([indices, indices + batch_size])
        return tensor[expanded]
    return tensor[indices]


def compute_x0_preds(
    *,
    pipeline,
    prev_latents: torch.Tensor,
    step_idx: int,
    t: torch.Tensor,
    callback_kwargs: Dict[str, torch.Tensor],
    eta: float,
) -> torch.Tensor:
    prompt_embeds = callback_kwargs["prompt_embeds"]
    latent_model_input = (
        torch.cat([prev_latents] * 2)
        if pipeline.do_classifier_free_guidance
        else prev_latents
    )
    latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

    added_cond_kwargs = None
    timestep_cond = None
    if isinstance(pipeline, StableDiffusionXLPipeline) or "XL" in pipeline.__class__.__name__:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mixed FK resampling + pruning search and log per-step rewards."
    )
    parser.add_argument("--prompts-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="stable-diffusion-v1-5")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--num-particles", type=int, default=32)
    parser.add_argument("--time-steps", type=int, default=64)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--fk-resample-steps", type=str, default="6,12,8,24,30")
    parser.add_argument("--mt-cutoffs", type=str, default="4,8,16,32")
    parser.add_argument("--mt-remaining", type=str, default="8,4,2,1")
    parser.add_argument("--num-fk-runs", type=int, default=10)
    parser.add_argument("--reward-name", type=str, default="ImageReward")
    parser.add_argument("--lmbda", type=float, default=2.0)
    parser.add_argument("--potential-type", type=str, default="max")
    parser.add_argument("--adaptive-resampling", action="store_true", default=True)
    parser.add_argument("--no-adaptive-resampling", action="store_false", dest="adaptive_resampling")

    args = parser.parse_args()

    if args.num_prompts < 1:
        raise ValueError("--num-prompts must be >= 1")
    if args.num_particles < 1:
        raise ValueError("--num-particles must be >= 1")
    if args.time_steps < 1:
        raise ValueError("--time-steps must be >= 1")
    if args.num_fk_runs < 1:
        raise ValueError("--num-fk-runs must be >= 1")

    fk_resample_steps = parse_int_list(args.fk_resample_steps)
    mt_cutoffs = parse_int_list(args.mt_cutoffs)
    mt_remaining = parse_int_list(args.mt_remaining)
    if len(mt_remaining) != len(mt_cutoffs):
        raise ValueError("--mt-remaining must match --mt-cutoffs length")

    cutoff_map = {step: keep for step, keep in zip(mt_cutoffs, mt_remaining)}
    fk_resample_set = set(fk_resample_steps)

    output_dir = os.path.join(args.output_dir, args.output_name)
    os.makedirs(output_dir, exist_ok=False)
    output_csv_path = os.path.join(output_dir, "fk_pruning_rewards.csv")
    output_plot_path = os.path.join(output_dir, "final_reward_hist.png")
    args_path = os.path.join(output_dir, "args.json")

    with open(args_path, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    prompts = read_prompts(args.prompts_path, args.num_prompts)
    if not prompts:
        raise ValueError("No prompts found in the provided prompts file.")

    pipeline = load_pipeline(args.model_name, args.device)
    pipeline.set_progress_bar_config(disable=True)

    callback_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    if isinstance(pipeline, StableDiffusionXLPipeline) or "XL" in pipeline.__class__.__name__:
        callback_inputs.extend(
            [
                "add_text_embeds",
                "negative_pooled_prompt_embeds",
                "add_time_ids",
                "negative_add_time_ids",
            ]
        )

    total_runs = len(prompts) * args.num_fk_runs
    final_rewards: List[float] = []

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile, tqdm(
        total=len(prompts), desc="Prompts", unit="prompt"
    ) as prompt_bar, tqdm(
        total=total_runs, desc="FK runs", unit="run", leave=False
    ) as run_bar:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "prompt_id",
                "prompt",
                "run_id",
                "seed",
                "step",
                "timestep",
                "reward",
            ],
        )
        writer.writeheader()

        for prompt_id, prompt in enumerate(prompts):
            for run_id in range(args.num_fk_runs):
                base_seed = args.seed + (prompt_id * args.num_fk_runs + run_id) * args.num_particles
                particle_seeds = [base_seed + idx for idx in range(args.num_particles)]
                generators = [
                    torch.Generator(device=args.device).manual_seed(seed)
                    for seed in particle_seeds
                ]

                torch.manual_seed(base_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(base_seed)

                state = {
                    "population_rs": torch.zeros(args.num_particles, device=args.device),
                    "product_of_potentials": torch.ones(args.num_particles, device=args.device),
                }

                reward_fn = build_reward_fn(
                    pipeline=pipeline, reward_name=args.reward_name, prompt=prompt
                )

                current_seeds = particle_seeds[:]
                current_rewards: Optional[torch.Tensor] = None
                prev_latents: Optional[torch.Tensor] = None

                def callback_on_step_end(_pipeline, step_idx, timestep_value, callback_kwargs):
                    nonlocal current_seeds, current_rewards, state, prev_latents

                    latents = callback_kwargs["latents"]
                    if prev_latents is None:
                        raise RuntimeError("prev_latents is not initialized")

                    x0_preds = compute_x0_preds(
                        pipeline=pipeline,
                        prev_latents=prev_latents,
                        step_idx=step_idx,
                        t=timestep_value,
                        callback_kwargs=callback_kwargs,
                        eta=args.eta,
                    )
                    rewards = reward_fn(x0_preds)

                    step_number = step_idx + 1
                    indices = None
                    batch_size_before = latents.shape[0]

                    if step_number in fk_resample_set:
                        latents, rewards, indices = fk_resample(
                            latents=latents,
                            rewards=rewards,
                            state=state,
                            lmbda=args.lmbda,
                            potential_type=args.potential_type,
                            adaptive_resampling=args.adaptive_resampling,
                            step_idx=step_idx,
                            time_steps=args.time_steps,
                        )
                        if indices is not None:
                            current_seeds = [current_seeds[i] for i in indices.tolist()]

                    if step_number in cutoff_map:
                        keep_k = max(1, min(int(cutoff_map[step_number]), len(current_seeds)))
                        state["population_rs"] = rewards
                        batch_size_before = latents.shape[0]
                        topk_indices = torch.topk(rewards, k=keep_k).indices
                        latents = latents[topk_indices]
                        rewards = rewards[topk_indices]
                        current_seeds = [current_seeds[i] for i in topk_indices.tolist()]
                        state["population_rs"] = state["population_rs"][topk_indices]
                        state["product_of_potentials"] = state["product_of_potentials"][topk_indices]

                        indices = topk_indices

                    current_rewards = rewards

                    for seed, reward in zip(current_seeds, rewards.detach().cpu().tolist()):
                        writer.writerow(
                            {
                                "prompt_id": prompt_id,
                                "prompt": prompt,
                                "run_id": run_id,
                                "seed": seed,
                                "step": step_number,
                                "timestep": int(timestep_value),
                                "reward": float(reward),
                            }
                        )

                    outputs = {"latents": latents}
                    if indices is not None:
                        for key in (
                            "prompt_embeds",
                            "negative_prompt_embeds",
                            "add_text_embeds",
                            "negative_pooled_prompt_embeds",
                            "add_time_ids",
                            "negative_add_time_ids",
                        ):
                            if key in callback_kwargs:
                                outputs[key] = apply_indices(
                                    callback_kwargs[key],
                                    indices,
                                    batch_size=batch_size_before,
                                )
                    prev_latents = outputs["latents"]
                    return outputs

                with torch.no_grad():
                    init_latents = pipeline.prepare_latents(
                        batch_size=args.num_particles,
                        num_channels_latents=pipeline.unet.config.in_channels,
                        height=pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                        width=pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                        dtype=pipeline.unet.dtype,
                        device=pipeline.device,
                        generator=generators,
                        latents=None,
                    )
                    prev_latents = init_latents
                    pipeline(
                        [prompt] * args.num_particles,
                        num_inference_steps=args.time_steps,
                        eta=args.eta,
                        generator=generators,
                        latents=init_latents,
                        output_type="latent",
                        callback_on_step_end=callback_on_step_end,
                        callback_on_step_end_tensor_inputs=callback_inputs,
                    )

                if current_rewards is not None and len(current_rewards) > 0:
                    final_rewards.append(float(current_rewards.max().item()))

                run_bar.update(1)

            prompt_bar.update(1)

    mean, sd, ci = mean_ci(final_rewards)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(final_rewards, bins=30, alpha=0.75)
    ax.set_title("Final reward distribution")
    ax.set_xlabel("Final ImageReward")
    ax.set_ylabel("Run count")
    ax.text(
        0.02,
        0.98,
        f"Mean: {mean:.4f} | SD: {sd:.4f} | CI: ±{ci:.4f}",
        transform=ax.transAxes,
        va="top",
    )
    fig.tight_layout()
    plt.savefig(output_plot_path, dpi=150)
    plt.close(fig)

    print(f"Saved CSV to {output_csv_path}")
    print(f"Saved plot to {output_plot_path}")


if __name__ == "__main__":
    main()
