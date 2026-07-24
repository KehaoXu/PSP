#!/usr/bin/env python3
"""Quickstart script for running PSP (Predetermined Schedule Pruning) online.

This performs *live* generation with pruning applied during the diffusion
process itself (as opposed to the offline replay-on-precomputed-trajectories
setup used to produce the paper's main tables). At each user-specified
timestep, the current batch of particles is collapsed to a predicted clean
image x0, scored with a reward model, and pruned down to the requested
number of survivors before continuing denoising.

Example (SDXL, 8 initial particles, prune to 4 at step 16 and to 1 at step 32
out of 64 total steps):

    python run_psp.py \\
      --model sdxl \\
      --prompt "a photo of a cow left of a stop sign" \\
      --num-inference-steps 64 \\
      --num-particles 8 \\
      --prune-at 16,32 \\
      --keep 4,1 \\
      --output-dir outputs/quickstart_sdxl
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_TO_IMAGE_DIR = Path(__file__).resolve().parent
FKD_DIR = TEXT_TO_IMAGE_DIR / "fkd_diffusers"
for p in (TEXT_TO_IMAGE_DIR, FKD_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fks_utils import get_model  # noqa: E402
from fkd_diffusers.fkd_pipeline_sd import latent_to_decode as latent_to_decode_sd  # noqa: E402
from fkd_diffusers.fkd_pipeline_sd3 import FKDStableDiffusion3  # noqa: E402
from fkd_diffusers.fkd_pipeline_sdxl import (  # noqa: E402
    FKDStableDiffusionXL,
    latent_to_decode as latent_to_decode_sdxl,
)
from fkd_diffusers.rewards import do_human_preference_score, do_image_reward  # noqa: E402


# Short names accepted by `fks_utils.get_model` for the DDIM-based pipelines.
GET_MODEL_NAME_BY_KEY = {
    "sd15": "stable-diffusion-v1-5",
    "sdxl": "stable-diffusion-xl",
}
SD35_REPO_ID = "stabilityai/stable-diffusion-3.5-large"
HF_REPO_ID_BY_KEY = {
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd35": SD35_REPO_ID,
}
DEFAULT_STEPS_BY_KEY = {"sd15": 64, "sdxl": 64, "sd35": 32}


def synchronize_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _ddim_alpha_prev(scheduler, prev_timestep: int, device: torch.device) -> torch.Tensor:
    if prev_timestep >= 0:
        return scheduler.alphas_cumprod[prev_timestep].to(device=device)
    final_alpha = getattr(scheduler, "final_alpha_cumprod", None)
    if final_alpha is None:
        final_alpha = torch.tensor(1.0, device=device, dtype=torch.float32)
    elif not isinstance(final_alpha, torch.Tensor):
        final_alpha = torch.tensor(final_alpha, device=device, dtype=torch.float32)
    else:
        final_alpha = final_alpha.to(device=device)
    return final_alpha


def estimate_x0_from_ddim_transition(
    *,
    scheduler,
    timestep: torch.Tensor,
    sample_before_step: torch.Tensor,
    sample_after_step: torch.Tensor,
    num_inference_steps: int,
) -> torch.Tensor:
    """Recovers x0 from a single DDIM transition (used for SD1.5 / SDXL)."""
    t_int = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
    step_ratio = scheduler.config.num_train_timesteps // num_inference_steps
    prev_t = t_int - step_ratio
    device = sample_before_step.device
    dtype = sample_before_step.dtype

    alpha_t = scheduler.alphas_cumprod[t_int].to(device=device, dtype=torch.float32)
    alpha_prev = _ddim_alpha_prev(scheduler, prev_t, device=device).to(dtype=torch.float32)

    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_alpha_prev = torch.sqrt(alpha_prev)
    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
    sqrt_one_minus_alpha_prev = torch.sqrt(1.0 - alpha_prev)

    c1 = sqrt_alpha_prev / sqrt_alpha_t
    c2 = sqrt_one_minus_alpha_prev - c1 * sqrt_one_minus_alpha_t
    if abs(float(c2)) < 1e-8:
        return sample_after_step.to(dtype=dtype)

    sample_before = sample_before_step.to(torch.float32)
    sample_after = sample_after_step.to(torch.float32)
    eps = (sample_after - c1 * sample_before) / c2
    x0 = (sample_before - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t
    return x0.to(dtype=dtype)


def estimate_velocity_from_flowmatch_transition(
    *,
    scheduler,
    timestep: torch.Tensor,
    sample_before_step: torch.Tensor,
    sample_after_step: torch.Tensor,
) -> torch.Tensor:
    if hasattr(scheduler, "index_for_timestep"):
        sigma_idx = scheduler.index_for_timestep(timestep)
    else:
        timesteps = scheduler.timesteps
        sigma_idx = int((timesteps == timestep).nonzero(as_tuple=True)[0][0].item())
    sigma = scheduler.sigmas[sigma_idx].to(
        device=sample_before_step.device, dtype=sample_before_step.dtype
    )
    sigma_next = scheduler.sigmas[sigma_idx + 1].to(
        device=sample_before_step.device, dtype=sample_before_step.dtype
    )
    delta = sigma_next - sigma
    if abs(float(delta.item())) < 1e-8:
        return torch.zeros_like(sample_before_step)
    return (sample_after_step - sample_before_step) / delta


def estimate_x0_from_sd35_transition(
    *,
    scheduler,
    timestep: torch.Tensor,
    sample_before_step: torch.Tensor,
    sample_after_step: torch.Tensor,
) -> torch.Tensor:
    """Recovers x0 from a single flow-matching Euler transition (SD3.5)."""
    velocity = estimate_velocity_from_flowmatch_transition(
        scheduler=scheduler,
        timestep=timestep,
        sample_before_step=sample_before_step,
        sample_after_step=sample_after_step,
    )
    if hasattr(scheduler, "index_for_timestep"):
        sigma_idx = scheduler.index_for_timestep(timestep)
    else:
        timesteps = scheduler.timesteps
        sigma_idx = int((timesteps == timestep).nonzero(as_tuple=True)[0][0].item())
    sigma = scheduler.sigmas[sigma_idx].to(
        device=sample_before_step.device, dtype=sample_before_step.dtype
    )
    while sigma.ndim < sample_before_step.ndim:
        sigma = sigma.view(*sigma.shape, 1)
    return sample_before_step - sigma * velocity


def decode_latents_to_tensor_sd35(*, pipeline, latents: torch.Tensor) -> torch.Tensor:
    vae = pipeline.vae
    needs_upcast = vae.dtype == torch.float16 and getattr(vae.config, "force_upcast", False)
    if needs_upcast:
        if hasattr(pipeline, "upcast_vae"):
            pipeline.upcast_vae()
        else:
            vae.to(dtype=torch.float32)
    with torch.no_grad():
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
    return image


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
    first_dim = int(tensor.shape[0]) if tensor.ndim > 0 else 0
    if first_dim == batch_size * 2:
        # Classifier-free-guidance tensors stack [uncond; cond] along dim 0.
        expanded = torch.cat([indices, indices + batch_size])
        return tensor[expanded]
    if first_dim == batch_size:
        return tensor[indices]
    return tensor


def parse_schedule(
    prune_at: str, keep: str, num_inference_steps: int, num_particles: int
) -> List[Tuple[int, int]]:
    steps = [int(s) for s in prune_at.split(",") if s.strip()] if prune_at else []
    keeps = [int(s) for s in keep.split(",") if s.strip()] if keep else []
    if len(steps) != len(keeps):
        raise ValueError("--prune-at and --keep must have the same number of comma-separated values")
    schedule = sorted(zip(steps, keeps), key=lambda x: x[0])
    for step, keep_count in schedule:
        if not (1 <= step <= num_inference_steps):
            raise ValueError(f"prune step {step} out of range [1, {num_inference_steps}]")
        if keep_count < 1 or keep_count > num_particles:
            raise ValueError(f"keep count {keep_count} out of range [1, {num_particles}]")
    if not schedule or schedule[-1][0] != num_inference_steps:
        # Always finalize down to a single winner at the last step unless the
        # caller already specified what to keep there.
        schedule.append((num_inference_steps, 1))
    return schedule


def build_pipeline(model_key: str, device: str):
    is_sd35 = model_key == "sd35"
    if is_sd35:
        pipeline = FKDStableDiffusion3.from_pretrained(SD35_REPO_ID, torch_dtype=torch.float16)
        # SD3.5 is large enough that CPU offload is needed on most single GPUs.
        pipeline.enable_model_cpu_offload(device=device)
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
    else:
        pipeline = get_model(GET_MODEL_NAME_BY_KEY[model_key]).to(device)
    return pipeline, is_sd35


def score_particles(
    *,
    pipeline,
    is_sd35: bool,
    reward_fn: str,
    image_tensor: torch.Tensor,
    prompts: Sequence[str],
) -> List[float]:
    if reward_fn == "ImageReward":
        scores = do_image_reward(prompts=list(prompts), image_tensors=image_tensor)
    elif reward_fn == "HumanPreference":
        pil_images = pipeline.image_processor.postprocess(image_tensor, output_type="pil")
        scores = do_human_preference_score(images=pil_images, prompts=list(prompts))
    else:
        raise ValueError(f"Unknown --reward-fn {reward_fn}")
    return [float(s) for s in scores]


def run_psp(
    *,
    pipeline,
    is_sd35: bool,
    prompt: str,
    device: str,
    seed: int,
    num_inference_steps: int,
    num_particles: int,
    schedule: List[Tuple[int, int]],
    reward_fn: str,
    eta: float,
) -> Tuple[torch.Tensor, List[Dict], float]:
    keep_by_step = dict(schedule)
    eval_set = set(keep_by_step.keys())

    prompt_list = [prompt] * num_particles
    generators = [torch.Generator(device=device).manual_seed(seed + i) for i in range(num_particles)]

    if is_sd35:
        prev_latents = pipeline.prepare_latents(
            batch_size=num_particles,
            num_channels_latents=pipeline.transformer.config.in_channels,
            height=pipeline.transformer.config.sample_size * pipeline.vae_scale_factor,
            width=pipeline.transformer.config.sample_size * pipeline.vae_scale_factor,
            dtype=pipeline.transformer.dtype,
            device=pipeline._execution_device,
            generator=list(generators),
            latents=None,
        )
    else:
        prev_latents = pipeline.prepare_latents(
            batch_size=num_particles,
            num_channels_latents=pipeline.unet.config.in_channels,
            height=pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
            width=pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
            dtype=pipeline.unet.dtype,
            device=pipeline.device,
            generator=generators,
            latents=None,
        )

    callback_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    if is_sd35:
        callback_inputs.extend(["pooled_prompt_embeds", "negative_pooled_prompt_embeds"])
    elif isinstance(pipeline, FKDStableDiffusionXL):
        callback_inputs.extend(
            ["add_text_embeds", "negative_pooled_prompt_embeds", "add_time_ids", "negative_add_time_ids"]
        )

    trace: List[Dict] = []

    def callback_on_step_end(_pipeline, step_idx, t, callback_kwargs):
        nonlocal prev_latents, prompt_list
        latents_after_step = callback_kwargs.get("latents", prev_latents)
        step_number = step_idx + 1
        if step_number not in eval_set:
            prev_latents = latents_after_step
            return {"latents": latents_after_step}

        if is_sd35:
            x0_preds = estimate_x0_from_sd35_transition(
                scheduler=pipeline.scheduler,
                timestep=t,
                sample_before_step=prev_latents,
                sample_after_step=latents_after_step,
            )
            image_tensor = decode_latents_to_tensor_sd35(pipeline=pipeline, latents=x0_preds).detach()
        else:
            x0_preds = estimate_x0_from_ddim_transition(
                scheduler=pipeline.scheduler,
                timestep=t,
                sample_before_step=prev_latents,
                sample_after_step=latents_after_step,
                num_inference_steps=num_inference_steps,
            )
            decode_fn = (
                latent_to_decode_sdxl if isinstance(pipeline, FKDStableDiffusionXL) else latent_to_decode_sd
            )
            image_tensor = decode_fn(model=pipeline, output_type="pil", latents=x0_preds).detach()

        rewards = score_particles(
            pipeline=pipeline,
            is_sd35=is_sd35,
            reward_fn=reward_fn,
            image_tensor=image_tensor,
            prompts=prompt_list,
        )
        rewards_t = torch.tensor(rewards, device=latents_after_step.device, dtype=torch.float32)
        keep_count = max(1, min(int(keep_by_step[step_number]), int(latents_after_step.shape[0])))

        batch_before = int(latents_after_step.shape[0])
        outputs = {"latents": latents_after_step}
        if keep_count < batch_before:
            order = torch.argsort(rewards_t, descending=True)
            indices = order[:keep_count]
            outputs["latents"] = latents_after_step[indices]
            for key in (
                "prompt_embeds",
                "negative_prompt_embeds",
                "pooled_prompt_embeds",
                "negative_pooled_prompt_embeds",
                "add_text_embeds",
                "add_time_ids",
                "negative_add_time_ids",
            ):
                if key in callback_kwargs:
                    outputs[key] = apply_indices(callback_kwargs[key], indices, batch_size=batch_before)
            prompt_list = [prompt] * keep_count

        trace.append(
            {
                "step": step_number,
                "batch_before": batch_before,
                "batch_after": int(outputs["latents"].shape[0]),
                "reward_mean": float(rewards_t.mean().item()),
                "reward_max": float(rewards_t.max().item()),
                "rewards": rewards,
            }
        )
        prev_latents = outputs["latents"]
        return outputs

    synchronize_if_needed(device)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        if is_sd35:
            out = pipeline(
                prompt=[prompt] * num_particles,
                num_inference_steps=num_inference_steps,
                generator=list(generators),
                latents=prev_latents,
                output_type="latent",
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_inputs,
            )
        else:
            out = pipeline(
                prompt=[prompt] * num_particles,
                num_inference_steps=num_inference_steps,
                eta=eta,
                generator=generators,
                latents=prev_latents,
                output_type="latent",
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_inputs,
            )
    synchronize_if_needed(device)
    elapsed_s = time.perf_counter() - t0
    peak_vram_gb = None
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    final_latents = out.images if hasattr(out, "images") else out[0]
    return final_latents, trace, elapsed_s, peak_vram_gb


def save_final_images(
    *, pipeline, is_sd35: bool, latents: torch.Tensor, output_dir: Path
) -> List[str]:
    if is_sd35:
        image_tensor = decode_latents_to_tensor_sd35(pipeline=pipeline, latents=latents).detach()
    else:
        decode_fn = (
            latent_to_decode_sdxl if isinstance(pipeline, FKDStableDiffusionXL) else latent_to_decode_sd
        )
        image_tensor = decode_fn(model=pipeline, output_type="pil", latents=latents).detach()
    pil_images = pipeline.image_processor.postprocess(image_tensor, output_type="pil")
    saved = []
    for i, image in enumerate(pil_images):
        p = output_dir / f"final_{i:02d}.png"
        image.save(p)
        saved.append(str(p))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PSP (predetermined-schedule particle pruning) online for a single prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, choices=sorted(HF_REPO_ID_BY_KEY.keys()))
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate.")
    parser.add_argument("--num-particles", type=int, default=8, help="Initial number of particles N.")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Total denoising steps (defaults to 64 for sd15/sdxl, 32 for sd35).",
    )
    parser.add_argument(
        "--prune-at",
        type=str,
        default="",
        help="Comma-separated timesteps at which to prune particles, e.g. '16,32'.",
    )
    parser.add_argument(
        "--keep",
        type=str,
        default="",
        help="Comma-separated particle counts to keep at each --prune-at step, e.g. '4,1'.",
    )
    parser.add_argument(
        "--reward-fn",
        type=str,
        default="ImageReward",
        choices=["ImageReward", "HumanPreference"],
        help="Reward model used to score predicted clean images (x0) at each pruning step.",
    )
    parser.add_argument("--eta", type=float, default=1.0, help="DDIM eta (sd15/sdxl only).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    num_inference_steps = args.num_inference_steps or DEFAULT_STEPS_BY_KEY[args.model]
    schedule = parse_schedule(args.prune_at, args.keep, num_inference_steps, args.num_particles)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_psp] model: {args.model} ({HF_REPO_ID_BY_KEY[args.model]})")
    print(f"[run_psp] prompt: {args.prompt!r}")
    print(f"[run_psp] num_inference_steps: {num_inference_steps}")
    print(f"[run_psp] num_particles: {args.num_particles}")
    print(f"[run_psp] schedule (step -> keep): {schedule}")
    print(f"[run_psp] reward_fn: {args.reward_fn}")

    pipeline, is_sd35 = build_pipeline(args.model, args.device)
    pipeline.set_progress_bar_config(disable=True)

    final_latents, trace, elapsed_s, peak_vram_gb = run_psp(
        pipeline=pipeline,
        is_sd35=is_sd35,
        prompt=args.prompt,
        device=args.device,
        seed=args.seed,
        num_inference_steps=num_inference_steps,
        num_particles=args.num_particles,
        schedule=schedule,
        reward_fn=args.reward_fn,
        eta=args.eta,
    )
    saved_images = save_final_images(
        pipeline=pipeline, is_sd35=is_sd35, latents=final_latents, output_dir=output_dir
    )

    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "seed": args.seed,
        "num_inference_steps": num_inference_steps,
        "num_particles": args.num_particles,
        "schedule": schedule,
        "reward_fn": args.reward_fn,
        "elapsed_s": elapsed_s,
        "peak_vram_gb": peak_vram_gb,
        "trace": trace,
        "saved_images": saved_images,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("")
    print("[run_psp] done")
    print(f"[run_psp] output dir: {output_dir}")
    print(f"[run_psp] wallclock: {elapsed_s:.2f}s")
    if peak_vram_gb is not None:
        print(f"[run_psp] peak VRAM: {peak_vram_gb:.2f} GB")
    for row in trace:
        print(
            f"[run_psp] step {row['step']:>3}: {row['batch_before']} -> {row['batch_after']} particles "
            f"(reward mean={row['reward_mean']:.3f}, max={row['reward_max']:.3f})"
        )
    print(f"[run_psp] saved {len(saved_images)} final image(s):")
    for p in saved_images:
        print(f"  - {p}")
    print("")


if __name__ == "__main__":
    main()
