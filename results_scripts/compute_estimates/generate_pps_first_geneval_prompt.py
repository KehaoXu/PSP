#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_TO_IMAGE_DIR = REPO_ROOT / "Fk-Diffusion-Steering" / "text_to_image"
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
from fkd_diffusers.rewards import do_image_reward  # noqa: E402


def _append_prompt(prompts: List[str], value) -> None:
    if isinstance(value, str):
        t = value.strip()
        if t:
            prompts.append(t)


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
    raise ValueError("This utility expects a .jsonl prompt file.")


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
        expanded = torch.cat([indices, indices + batch_size])
        return tensor[expanded]
    if first_dim == batch_size:
        return tensor[indices]
    return tensor


def compute_checkpoint_steps(total_steps: int) -> List[int]:
    s25 = max(1, min(total_steps, int(round(0.25 * total_steps))))
    s50 = max(s25 + 1, min(total_steps, int(round(0.50 * total_steps))))
    return [s25, s50, total_steps]


def build_pipeline(model_name: str, device: str):
    normalized = model_name.strip().lower()
    is_sd35 = normalized in {
        "stable-diffusion-3.5-large",
        "stable-diffusion-v3-5",
        "stable-diffusion-3.5",
        "sd3.5",
    }
    if is_sd35:
        pipeline = FKDStableDiffusion3.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float16
        ).to(device)
        return pipeline, True
    return get_model(model_name).to(device), False


def run_pps_first_prompt(
    *,
    pipeline,
    is_sd35: bool,
    prompt: str,
    device: str,
    seed: int,
    time_steps: int,
    eta: float,
):
    pps_steps = compute_checkpoint_steps(time_steps)
    keep_by_step = {pps_steps[0]: 4, pps_steps[1]: 2, pps_steps[2]: 2}
    eval_set = set(pps_steps)

    init_particles = 8
    prompt_list = [prompt] * init_particles
    generators = [torch.Generator(device=device).manual_seed(seed + i) for i in range(init_particles)]

    if is_sd35:
        prev_latents = pipeline.prepare_latents(
            batch_size=init_particles,
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
            batch_size=init_particles,
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
            [
                "add_text_embeds",
                "negative_pooled_prompt_embeds",
                "add_time_ids",
                "negative_add_time_ids",
            ]
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
            image_tensor = decode_latents_to_tensor_sd35(
                pipeline=pipeline, latents=x0_preds
            ).detach()
        else:
            x0_preds = estimate_x0_from_ddim_transition(
                scheduler=pipeline.scheduler,
                timestep=t,
                sample_before_step=prev_latents,
                sample_after_step=latents_after_step,
                num_inference_steps=time_steps,
            )
            decode_fn = (
                latent_to_decode_sdxl
                if isinstance(pipeline, FKDStableDiffusionXL)
                else latent_to_decode_sd
            )
            image_tensor = decode_fn(model=pipeline, output_type="pil", latents=x0_preds).detach()

        scores = do_image_reward(prompts=prompt_list, image_tensors=image_tensor)
        rewards = torch.tensor(scores, device=latents_after_step.device, dtype=torch.float32)
        keep_count = max(1, min(int(keep_by_step[step_number]), int(latents_after_step.shape[0])))

        batch_before = int(latents_after_step.shape[0])
        outputs = {"latents": latents_after_step}
        if keep_count < batch_before:
            order = torch.argsort(rewards, descending=True)
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
                    outputs[key] = apply_indices(
                        callback_kwargs[key], indices, batch_size=batch_before
                    )
            prompt_list = [prompt] * keep_count
        trace.append(
            {
                "step": step_number,
                "batch_before": batch_before,
                "batch_after": int(outputs["latents"].shape[0]),
                "reward_mean": float(rewards.mean().item()),
                "reward_max": float(rewards.max().item()),
            }
        )
        prev_latents = outputs["latents"]
        return outputs

    synchronize_if_needed(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        if is_sd35:
            out = pipeline(
                prompt=[prompt] * init_particles,
                num_inference_steps=time_steps,
                generator=list(generators),
                latents=prev_latents,
                output_type="latent",
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_inputs,
            )
        else:
            out = pipeline(
                prompt=[prompt] * init_particles,
                num_inference_steps=time_steps,
                eta=eta,
                generator=generators,
                latents=prev_latents,
                output_type="latent",
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_inputs,
            )
    synchronize_if_needed(device)
    elapsed_s = time.perf_counter() - t0
    final_latents = out.images if hasattr(out, "images") else out[0]
    return final_latents, trace, elapsed_s, pps_steps


def save_final_images(*, pipeline, is_sd35: bool, latents: torch.Tensor, output_dir: Path) -> List[str]:
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
        description="Generate first Geneval prompt with PPS 8->4->2 and save final 2 images."
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument(
        "--prompts-path",
        type=str,
        default=str(TEXT_TO_IMAGE_DIR / "prompt_files" / "geneval_metadata.jsonl"),
    )
    parser.add_argument("--prompt-start-id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-steps", type=int, required=True)
    parser.add_argument("--eta", type=float, default=0.0)
    args = parser.parse_args()

    prompts = read_prompts(args.prompts_path)
    if args.prompt_start_id < 0 or args.prompt_start_id >= len(prompts):
        raise ValueError("prompt_start_id out of range")
    prompt = prompts[args.prompt_start_id]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    samples_dir = output_dir / "final_samples"
    samples_dir.mkdir(parents=True, exist_ok=False)

    pipeline, is_sd35 = build_pipeline(args.model_name, args.device)
    pipeline.set_progress_bar_config(disable=True)

    final_latents, trace, elapsed_s, pps_steps = run_pps_first_prompt(
        pipeline=pipeline,
        is_sd35=is_sd35,
        prompt=prompt,
        device=args.device,
        seed=args.seed,
        time_steps=args.time_steps,
        eta=args.eta,
    )
    saved_images = save_final_images(
        pipeline=pipeline, is_sd35=is_sd35, latents=final_latents, output_dir=samples_dir
    )

    payload = {
        "model_name": args.model_name,
        "device": args.device,
        "prompt_start_id": args.prompt_start_id,
        "prompt": prompt,
        "seed": args.seed,
        "time_steps": args.time_steps,
        "pps_steps": pps_steps,
        "elapsed_s": elapsed_s,
        "trace": trace,
        "saved_images": saved_images,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("")
    print("[pps-validate] done")
    print(f"[pps-validate] output: {output_dir}")
    print(f"[pps-validate] elapsed_s: {elapsed_s:.4f}")
    print(f"[pps-validate] pps_steps: {pps_steps}")
    print(f"[pps-validate] saved_images: {len(saved_images)}")
    for p in saved_images:
        print(f"  - {p}")
    print("")


if __name__ == "__main__":
    main()
