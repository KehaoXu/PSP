#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_TO_IMAGE_DIR = REPO_ROOT / "Fk-Diffusion-Steering" / "text_to_image"
FKD_DIR = TEXT_TO_IMAGE_DIR / "fkd_diffusers"
for p in (TEXT_TO_IMAGE_DIR, FKD_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fks_utils import do_eval, get_model  # noqa: E402
from fkd_diffusers.fkd_pipeline_sd import (  # noqa: E402
    latent_to_decode as latent_to_decode_sd,
)
from fkd_diffusers.fkd_pipeline_sd3 import FKDStableDiffusion3  # noqa: E402
from fkd_diffusers.fkd_pipeline_sdxl import (  # noqa: E402
    FKDStableDiffusionXL,
    latent_to_decode as latent_to_decode_sdxl,
)


def _append_prompt(prompts: List[str], value) -> None:
    if not isinstance(value, str):
        return
    text = value.strip()
    if text:
        prompts.append(text)


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


def synchronize_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_sync(device: str) -> float:
    t0 = time.perf_counter()
    synchronize_if_needed(device)
    return time.perf_counter() - t0


def cuda_memory_snapshot(device: str) -> Dict[str, float]:
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return {
            "num_alloc_retries": 0.0,
            "allocated_bytes_all_peak": 0.0,
            "reserved_bytes_all_peak": 0.0,
        }
    stats = torch.cuda.memory_stats()
    return {
        "num_alloc_retries": float(stats.get("num_alloc_retries", 0.0)),
        "allocated_bytes_all_peak": float(stats.get("allocated_bytes.all.peak", 0.0)),
        "reserved_bytes_all_peak": float(stats.get("reserved_bytes.all.peak", 0.0)),
    }


def memory_delta(after: Dict[str, float], before: Dict[str, float], key: str) -> float:
    return float(after.get(key, 0.0) - before.get(key, 0.0))


def set_cudnn_benchmark(mode: str) -> str:
    if mode == "auto":
        return f"unchanged:{torch.backends.cudnn.benchmark}"
    enabled = mode == "on"
    torch.backends.cudnn.benchmark = enabled
    return f"set:{enabled}"


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


def compute_checkpoint_steps(total_steps: int, fractions: Sequence[float]) -> List[int]:
    raw_steps: List[int] = []
    for frac in fractions:
        step = int(round(frac * total_steps))
        step = max(1, min(total_steps, step))
        raw_steps.append(step)
    if raw_steps[-1] != total_steps:
        raw_steps[-1] = total_steps
    for i in range(1, len(raw_steps)):
        raw_steps[i] = max(raw_steps[i], raw_steps[i - 1] + 1)
    raw_steps[-1] = total_steps
    if any(s > total_steps for s in raw_steps):
        raise ValueError("Invalid checkpoint steps generated")
    return raw_steps


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def run_single_method(
    *,
    pipeline,
    is_sd35: bool,
    prompt: str,
    method: str,
    prompt_id: int,
    seed_base: int,
    device: str,
    time_steps: int,
    eta: float,
    eval_steps: List[int],
    keep_by_eval_step: Dict[int, int],
    eval_label_by_step: Dict[int, str],
    bon_init_particles: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    if method == "bon":
        init_particles = bon_init_particles
    elif method == "pps":
        init_particles = 8
    else:
        raise ValueError(f"Unknown method: {method}")

    prompt_list: List[str] = [prompt] * init_particles
    generators = [
        torch.Generator(device=device).manual_seed(seed_base + i)
        for i in range(init_particles)
    ]

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

    callback_total_s = 0.0
    decode_total_s = 0.0
    reward_total_s = 0.0
    per_eval_rows: List[Dict[str, float]] = []
    eval_step_set = set(eval_steps)

    callback_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    if is_sd35:
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

    def callback_on_step_end(_pipeline, step_idx, t, callback_kwargs):
        nonlocal prev_latents, callback_total_s, decode_total_s, reward_total_s, prompt_list
        latents_after_step = callback_kwargs.get("latents", prev_latents)
        step_number = step_idx + 1

        if step_number not in eval_step_set:
            prev_latents = latents_after_step
            return {"latents": latents_after_step}

        sync_pre_eval_s = timed_sync(device)
        callback_start = time.perf_counter()
        mem_before = cuda_memory_snapshot(device)
        use_cuda_timing = device.startswith("cuda") and torch.cuda.is_available()
        callback_gpu_ms = 0.0
        if use_cuda_timing:
            callback_event_start = torch.cuda.Event(enable_timing=True)
            callback_event_end = torch.cuda.Event(enable_timing=True)
            callback_event_start.record()
            x0_event_start = torch.cuda.Event(enable_timing=True)
            x0_event_end = torch.cuda.Event(enable_timing=True)
            x0_event_start.record()

        if is_sd35:
            x0_preds = estimate_x0_from_sd35_transition(
                scheduler=pipeline.scheduler,
                timestep=t,
                sample_before_step=prev_latents,
                sample_after_step=latents_after_step,
            )
        else:
            x0_preds = estimate_x0_from_ddim_transition(
                scheduler=pipeline.scheduler,
                timestep=t,
                sample_before_step=prev_latents,
                sample_after_step=latents_after_step,
                num_inference_steps=time_steps,
            )
        if use_cuda_timing:
            x0_event_end.record()
            synchronize_if_needed(device)
            x0_estimate_gpu_ms = float(x0_event_start.elapsed_time(x0_event_end))
            decode_event_start = torch.cuda.Event(enable_timing=True)
            decode_event_end = torch.cuda.Event(enable_timing=True)
            decode_event_start.record()
        else:
            x0_estimate_gpu_ms = 0.0

        decode_start = time.perf_counter()
        if is_sd35:
            image_tensor = decode_latents_to_tensor_sd35(
                pipeline=pipeline, latents=x0_preds
            ).detach()
        else:
            decode_fn = (
                latent_to_decode_sdxl
                if isinstance(pipeline, FKDStableDiffusionXL)
                else latent_to_decode_sd
            )
            image_tensor = decode_fn(
                model=pipeline, output_type="pil", latents=x0_preds
            ).detach()
        synchronize_if_needed(device)
        decode_elapsed = time.perf_counter() - decode_start
        if use_cuda_timing:
            decode_event_end.record()
            synchronize_if_needed(device)
            decode_gpu_ms = float(decode_event_start.elapsed_time(decode_event_end))
            reward_event_start = torch.cuda.Event(enable_timing=True)
            reward_event_end = torch.cuda.Event(enable_timing=True)
            reward_event_start.record()
        else:
            decode_gpu_ms = 0.0

        reward_start = time.perf_counter()
        eval_out = do_eval(
            prompt=prompt_list,
            images=image_tensor,
            metrics_to_compute=["ImageReward"],
        )
        synchronize_if_needed(device)
        reward_elapsed = time.perf_counter() - reward_start
        if use_cuda_timing:
            reward_event_end.record()
            synchronize_if_needed(device)
            reward_gpu_ms = float(reward_event_start.elapsed_time(reward_event_end))
        else:
            reward_gpu_ms = 0.0

        rewards = torch.tensor(
            eval_out["ImageReward"]["result"],
            device=latents_after_step.device,
            dtype=torch.float32,
        )

        batch_before = int(latents_after_step.shape[0])
        outputs = {"latents": latents_after_step}
        keep_count = int(keep_by_eval_step.get(step_number, batch_before))
        keep_count = max(1, min(batch_before, keep_count))
        batch_after = batch_before
        if keep_count < batch_before:
            if use_cuda_timing:
                ranking_event_start = torch.cuda.Event(enable_timing=True)
                ranking_event_end = torch.cuda.Event(enable_timing=True)
                ranking_event_start.record()
            order = torch.argsort(rewards, descending=True)
            indices = order[:keep_count]
            if use_cuda_timing:
                ranking_event_end.record()
                synchronize_if_needed(device)
                ranking_gpu_ms = float(ranking_event_start.elapsed_time(ranking_event_end))
                prune_event_start = torch.cuda.Event(enable_timing=True)
                prune_event_end = torch.cuda.Event(enable_timing=True)
                prune_event_start.record()
            else:
                ranking_gpu_ms = 0.0
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
            batch_after = keep_count
            if use_cuda_timing:
                prune_event_end.record()
                synchronize_if_needed(device)
                prune_gpu_ms = float(prune_event_start.elapsed_time(prune_event_end))
            else:
                prune_gpu_ms = 0.0
        else:
            ranking_gpu_ms = 0.0
            prune_gpu_ms = 0.0

        synchronize_if_needed(device)
        callback_elapsed = time.perf_counter() - callback_start
        if use_cuda_timing:
            callback_event_end.record()
            synchronize_if_needed(device)
            callback_gpu_ms = float(callback_event_start.elapsed_time(callback_event_end))
        mem_after = cuda_memory_snapshot(device)
        alloc_retries_delta = memory_delta(mem_after, mem_before, "num_alloc_retries")
        allocated_peak_delta = memory_delta(mem_after, mem_before, "allocated_bytes_all_peak")
        reserved_peak_delta = memory_delta(mem_after, mem_before, "reserved_bytes_all_peak")
        other_elapsed = max(0.0, callback_elapsed - decode_elapsed - reward_elapsed)

        callback_total_s += callback_elapsed
        decode_total_s += decode_elapsed
        reward_total_s += reward_elapsed
        per_eval_rows.append(
            {
                "prompt_id": prompt_id,
                "method": method,
                "eval_step": step_number,
                "eval_label": eval_label_by_step[step_number],
                "batch_before": batch_before,
                "batch_after": batch_after,
                "pre_callback_drain_s": sync_pre_eval_s,
                "eval_total_s": callback_elapsed,
                "callback_gpu_ms": callback_gpu_ms,
                "x0_estimate_gpu_ms": x0_estimate_gpu_ms,
                "decode_vae_s": decode_elapsed,
                "decode_vae_gpu_ms": decode_gpu_ms,
                "reward_eval_s": reward_elapsed,
                "reward_eval_gpu_ms": reward_gpu_ms,
                "ranking_gpu_ms": ranking_gpu_ms,
                "prune_gpu_ms": prune_gpu_ms,
                "alloc_retries_delta": alloc_retries_delta,
                "allocated_peak_bytes_delta": allocated_peak_delta,
                "reserved_peak_bytes_delta": reserved_peak_delta,
                "eval_other_minus_pre_drain_s": max(0.0, other_elapsed - sync_pre_eval_s),
                "eval_other_s": other_elapsed,
            }
        )
        prev_latents = outputs["latents"]
        return outputs

    synchronize_if_needed(device)
    run_start = time.perf_counter()
    with torch.no_grad():
        if is_sd35:
            pipeline(
                prompt=[prompt] * init_particles,
                num_inference_steps=time_steps,
                generator=list(generators),
                latents=prev_latents,
                output_type="latent",
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_inputs,
            )
        else:
            pipeline(
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
    total_s = time.perf_counter() - run_start
    diffusion_s = max(0.0, total_s - callback_total_s)
    eval_other_total_s = max(0.0, callback_total_s - decode_total_s - reward_total_s)

    row = {
        "prompt_id": prompt_id,
        "method": method,
        "total_wall_s": total_s,
        "diffusion_s": diffusion_s,
        "eval_total_s": callback_total_s,
        "decode_vae_s": decode_total_s,
        "reward_eval_s": reward_total_s,
        "eval_other_s": eval_other_total_s,
        "eval_calls": float(len(per_eval_rows)),
    }
    return row, per_eval_rows


def build_pipeline(model_name: str, device: str):
    normalized = model_name.strip().lower()
    is_sd35 = normalized in {
        "stable-diffusion-3.5-large",
        "stable-diffusion-v3-5",
        "stable-diffusion-3.5",
        "sd3.5",
    }
    if is_sd35:
        # Force fp16 to avoid mixed fp16/bf16 prompt-encoder paths on some hosts.
        sd35_dtype = torch.float16
        pipeline = FKDStableDiffusion3.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large", torch_dtype=sd35_dtype
        ).to(device)
        return pipeline, True

    pipeline = get_model(model_name).to(device)
    return pipeline, False


def aggregate_method_rows(rows: List[Dict[str, float]], key: str) -> Dict[str, float]:
    values = [float(r[key]) for r in rows]
    mean_v, std_v = mean_std(values)
    return {"mean": mean_v, "std": std_v}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare wall-clock and evaluation overhead for Best-of-N vs PPS."
    )
    parser.add_argument(
        "--prompts-path",
        type=str,
        default=str(TEXT_TO_IMAGE_DIR / "prompt_files" / "benchmark_ir.json"),
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-start-id", type=int, default=0)
    parser.add_argument("--num-prompts-total", type=int, default=11)
    parser.add_argument("--num-prompts-stats", type=int, default=10)
    parser.add_argument("--time-steps", type=int, required=True)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument(
        "--methods",
        type=str,
        default="bon,pps",
        help="Comma-separated methods to run (subset of: bon,pps).",
    )
    parser.add_argument(
        "--bon-init-particles",
        type=int,
        default=4,
        help="Initial particle count for BoN method.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
    )
    args = parser.parse_args()

    if args.num_prompts_total < 2:
        raise ValueError("--num-prompts-total must be >= 2")
    if args.num_prompts_stats < 1:
        raise ValueError("--num-prompts-stats must be >= 1")
    if args.num_prompts_stats > args.num_prompts_total - 1:
        raise ValueError("--num-prompts-stats must be <= num-prompts-total - 1")
    if args.time_steps < 1:
        raise ValueError("--time-steps must be >= 1")
    if args.bon_init_particles < 1:
        raise ValueError("--bon-init-particles must be >= 1")
    cudnn_benchmark_state = set_cudnn_benchmark(args.cudnn_benchmark)
    requested_methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    if not requested_methods:
        raise ValueError("--methods must include at least one method")
    valid_methods = {"bon", "pps"}
    unknown_methods = [m for m in requested_methods if m not in valid_methods]
    if unknown_methods:
        raise ValueError(f"Unknown methods in --methods: {unknown_methods}")
    selected_methods = list(dict.fromkeys(requested_methods))

    all_prompts = read_prompts(args.prompts_path)
    if args.prompt_start_id < 0 or args.prompt_start_id >= len(all_prompts):
        raise ValueError("prompt start id out of range")
    selected_prompts = all_prompts[
        args.prompt_start_id : args.prompt_start_id + args.num_prompts_total
    ]
    if len(selected_prompts) != args.num_prompts_total:
        raise ValueError("not enough prompts for requested range")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    args_path = output_dir / "args.json"
    per_prompt_csv = output_dir / "per_prompt.csv"
    per_eval_csv = output_dir / "per_eval_instance.csv"
    summary_json = output_dir / "summary.json"

    pps_eval_steps = compute_checkpoint_steps(args.time_steps, [0.25, 0.5, 1.0])
    bon_eval_steps = [args.time_steps]
    pps_keep_by_step = {
        pps_eval_steps[0]: 4,
        pps_eval_steps[1]: 2,
        pps_eval_steps[2]: 2,
    }
    pps_labels = {
        pps_eval_steps[0]: "25pct",
        pps_eval_steps[1]: "50pct",
        pps_eval_steps[2]: "end",
    }
    bon_labels = {args.time_steps: "end"}

    args_payload = vars(args).copy()
    args_payload["method_configs"] = {
        "bon": {"init_particles": args.bon_init_particles, "eval_steps": bon_eval_steps},
        "pps": {
            "init_particles": 8,
            "eval_steps": pps_eval_steps,
            "keep_by_step": pps_keep_by_step,
        },
    }
    args_payload["selected_methods"] = selected_methods
    with open(args_path, "w", encoding="utf-8") as handle:
        json.dump(args_payload, handle, indent=2)

    pipeline, is_sd35 = build_pipeline(args.model_name, args.device)
    pipeline.set_progress_bar_config(disable=True)

    per_prompt_rows: List[Dict[str, float]] = []
    per_eval_rows: List[Dict[str, float]] = []

    total_items = len(selected_prompts) * len(selected_methods)
    with tqdm(total=total_items, desc="BoN vs PPS", unit="run") as pbar:
        for local_idx, prompt in enumerate(selected_prompts):
            prompt_id = args.prompt_start_id + local_idx
            prompt_seed_base = args.seed + prompt_id * 100
            for method in selected_methods:
                if method == "bon":
                    row, eval_rows = run_single_method(
                        pipeline=pipeline,
                        is_sd35=is_sd35,
                        prompt=prompt,
                        method=method,
                        prompt_id=prompt_id,
                        seed_base=prompt_seed_base,
                        device=args.device,
                        time_steps=args.time_steps,
                        eta=args.eta,
                        eval_steps=bon_eval_steps,
                        keep_by_eval_step={args.time_steps: args.bon_init_particles},
                        eval_label_by_step=bon_labels,
                        bon_init_particles=args.bon_init_particles,
                    )
                else:
                    row, eval_rows = run_single_method(
                        pipeline=pipeline,
                        is_sd35=is_sd35,
                        prompt=prompt,
                        method=method,
                        prompt_id=prompt_id,
                        seed_base=prompt_seed_base,
                        device=args.device,
                        time_steps=args.time_steps,
                        eta=args.eta,
                        eval_steps=pps_eval_steps,
                        keep_by_eval_step=pps_keep_by_step,
                        eval_label_by_step=pps_labels,
                        bon_init_particles=args.bon_init_particles,
                    )
                per_prompt_rows.append(row)
                per_eval_rows.extend(eval_rows)
                pbar.update(1)

    with open(per_prompt_csv, "w", newline="", encoding="utf-8") as handle:
        fields = [
            "prompt_id",
            "method",
            "total_wall_s",
            "diffusion_s",
            "eval_total_s",
            "decode_vae_s",
            "reward_eval_s",
            "eval_other_s",
            "eval_calls",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in per_prompt_rows:
            writer.writerow(row)

    with open(per_eval_csv, "w", newline="", encoding="utf-8") as handle:
        fields = [
            "prompt_id",
            "method",
            "eval_step",
            "eval_label",
            "batch_before",
            "batch_after",
            "pre_callback_drain_s",
            "eval_total_s",
            "callback_gpu_ms",
            "x0_estimate_gpu_ms",
            "decode_vae_s",
            "decode_vae_gpu_ms",
            "reward_eval_s",
            "reward_eval_gpu_ms",
            "ranking_gpu_ms",
            "prune_gpu_ms",
            "alloc_retries_delta",
            "allocated_peak_bytes_delta",
            "reserved_peak_bytes_delta",
            "eval_other_minus_pre_drain_s",
            "eval_other_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in per_eval_rows:
            writer.writerow(row)

    dropped_prompt_ids = [
        args.prompt_start_id + i
        for i in range(args.num_prompts_total - args.num_prompts_stats)
    ]
    prompt_ids_used = set(
        args.prompt_start_id + i
        for i in range(args.num_prompts_total - args.num_prompts_stats, args.num_prompts_total)
    )
    filtered_prompt_rows = [r for r in per_prompt_rows if int(r["prompt_id"]) in prompt_ids_used]
    filtered_eval_rows = [r for r in per_eval_rows if int(r["prompt_id"]) in prompt_ids_used]

    method_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in selected_methods:
        rows = [r for r in filtered_prompt_rows if r["method"] == method]
        method_summary[method] = {
            "total_wall_s": aggregate_method_rows(rows, "total_wall_s"),
            "diffusion_s": aggregate_method_rows(rows, "diffusion_s"),
            "eval_total_s": aggregate_method_rows(rows, "eval_total_s"),
            "decode_vae_s": aggregate_method_rows(rows, "decode_vae_s"),
            "reward_eval_s": aggregate_method_rows(rows, "reward_eval_s"),
            "eval_other_s": aggregate_method_rows(rows, "eval_other_s"),
        }

    per_eval_instance_summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    eval_groups: List[Tuple[str, str]] = []
    if "bon" in selected_methods:
        eval_groups.append(("bon", "end"))
    if "pps" in selected_methods:
        eval_groups.extend([("pps", "25pct"), ("pps", "50pct"), ("pps", "end")])
    for method, label in eval_groups:
        group_rows = [
            r for r in filtered_eval_rows if r["method"] == method and r["eval_label"] == label
        ]
        per_eval_instance_summary.setdefault(method, {})
        per_eval_instance_summary[method][label] = {
            "eval_total_s": aggregate_method_rows(group_rows, "eval_total_s"),
            "decode_vae_s": aggregate_method_rows(group_rows, "decode_vae_s"),
            "reward_eval_s": aggregate_method_rows(group_rows, "reward_eval_s"),
            "eval_other_s": aggregate_method_rows(group_rows, "eval_other_s"),
        }

    gpu_vs_wall_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    pre_drain_summary: Dict[str, Dict[str, float]] = {}
    allocator_delta_summary: Dict[str, Dict[str, float]] = {}
    for label in ("25pct", "50pct", "end"):
        if "pps" not in selected_methods:
            break
        rows = [r for r in filtered_eval_rows if r["method"] == "pps" and r["eval_label"] == label]
        if not rows:
            continue
        gpu_vs_wall_summary[label] = {
            "callback": {
                "wall_ms_mean": aggregate_method_rows(rows, "eval_total_s")["mean"] * 1000.0,
                "gpu_ms_mean": aggregate_method_rows(rows, "callback_gpu_ms")["mean"],
            },
            "decode": {
                "wall_ms_mean": aggregate_method_rows(rows, "decode_vae_s")["mean"] * 1000.0,
                "gpu_ms_mean": aggregate_method_rows(rows, "decode_vae_gpu_ms")["mean"],
            },
            "reward": {
                "wall_ms_mean": aggregate_method_rows(rows, "reward_eval_s")["mean"] * 1000.0,
                "gpu_ms_mean": aggregate_method_rows(rows, "reward_eval_gpu_ms")["mean"],
            },
        }
        pre_drain_summary[label] = {
            "pre_callback_drain_ms_mean": aggregate_method_rows(rows, "pre_callback_drain_s")["mean"] * 1000.0,
            "eval_other_ms_mean": aggregate_method_rows(rows, "eval_other_s")["mean"] * 1000.0,
            "eval_other_minus_pre_drain_ms_mean": aggregate_method_rows(
                rows, "eval_other_minus_pre_drain_s"
            )["mean"]
            * 1000.0,
        }
        allocator_delta_summary[label] = {
            "alloc_retries_delta_mean": aggregate_method_rows(rows, "alloc_retries_delta")["mean"],
            "allocated_peak_bytes_delta_mean": aggregate_method_rows(rows, "allocated_peak_bytes_delta")["mean"],
            "reserved_peak_bytes_delta_mean": aggregate_method_rows(rows, "reserved_peak_bytes_delta")["mean"],
        }

    summary = {
        "config": {
            "model_name": args.model_name,
            "device": args.device,
            "time_steps": args.time_steps,
            "prompts_path": args.prompts_path,
            "prompt_start_id": args.prompt_start_id,
            "num_prompts_total": args.num_prompts_total,
            "num_prompts_stats": args.num_prompts_stats,
            "dropped_warmup_prompt_ids": dropped_prompt_ids,
            "prompt_ids_used_for_stats": sorted(prompt_ids_used),
            "pps_steps": pps_eval_steps,
            "bon_steps": bon_eval_steps,
            "cudnn_benchmark_arg": args.cudnn_benchmark,
            "cudnn_benchmark_state": cudnn_benchmark_state,
        },
        "method_level_summary": method_summary,
        "per_eval_instance_summary": per_eval_instance_summary,
        "gpu_vs_wall_summary": gpu_vs_wall_summary,
        "pre_drain_summary": pre_drain_summary,
        "allocator_delta_summary": allocator_delta_summary,
        "per_prompt_rows": per_prompt_rows,
        "per_eval_rows": per_eval_rows,
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("")
    print("[bon-vs-pps] config")
    print(f"  model_name={args.model_name}")
    print(f"  time_steps={args.time_steps}")
    print(f"  prompts_total={args.num_prompts_total}")
    print(f"  prompts_used_for_stats={args.num_prompts_stats}")
    print(f"  dropped_warmup_prompt_ids={dropped_prompt_ids}")
    print(f"  bon_eval_steps={bon_eval_steps}")
    print(f"  pps_eval_steps={pps_eval_steps}")
    print(f"  selected_methods={selected_methods}")
    print(f"  bon_init_particles={args.bon_init_particles}")
    print(f"  cudnn_benchmark={args.cudnn_benchmark}")
    print(f"  cudnn_benchmark_state={cudnn_benchmark_state}")
    print("")
    print("[bon-vs-pps] method-level wall clock and overhead (mean +/- sd over prompts)")
    print("[bon-vs-pps] done")
    print(f"[bon-vs-pps] per_prompt: {per_prompt_csv}")
    print(f"[bon-vs-pps] per_eval:   {per_eval_csv}")
    print(f"[bon-vs-pps] summary:    {summary_json}")
    print("")
    for method in selected_methods:
        stats = method_summary[method]
        print(
            f"[bon-vs-pps] {method} total_wall_s={stats['total_wall_s']['mean']:.4f} +/- {stats['total_wall_s']['std']:.4f}"
        )
        print(
            f"  diffusion_s={stats['diffusion_s']['mean']:.4f} +/- {stats['diffusion_s']['std']:.4f}"
        )
        print(
            f"  eval_total_s={stats['eval_total_s']['mean']:.4f} +/- {stats['eval_total_s']['std']:.4f}"
        )
        print(
            f"  decode_vae_s={stats['decode_vae_s']['mean']:.4f} +/- {stats['decode_vae_s']['std']:.4f}"
        )
        print(
            f"  reward_eval_s={stats['reward_eval_s']['mean']:.4f} +/- {stats['reward_eval_s']['std']:.4f}"
        )
        print(
            f"  eval_other_s={stats['eval_other_s']['mean']:.4f} +/- {stats['eval_other_s']['std']:.4f}"
        )
    print("")
    print("[bon-vs-pps] per-eval-instance overhead (mean +/- sd)")
    ordered_labels = {"bon": ["end"], "pps": ["25pct", "50pct", "end"]}
    for method in selected_methods:
        for label in ordered_labels.get(method, []):
            stats = per_eval_instance_summary[method][label]
            print(f"[bon-vs-pps] {method}:{label}")
            print(
                f"  eval_total_s={stats['eval_total_s']['mean']:.4f} +/- {stats['eval_total_s']['std']:.4f}"
            )
            print(
                f"  decode_vae_s={stats['decode_vae_s']['mean']:.4f} +/- {stats['decode_vae_s']['std']:.4f}"
            )
            print(
                f"  reward_eval_s={stats['reward_eval_s']['mean']:.4f} +/- {stats['reward_eval_s']['std']:.4f}"
            )
            print(
                f"  eval_other_s={stats['eval_other_s']['mean']:.4f} +/- {stats['eval_other_s']['std']:.4f}"
            )
    if gpu_vs_wall_summary:
        print("")
        print("[bon-vs-pps] pps gpu-vs-wall summary (ms means)")
        for label, payload in gpu_vs_wall_summary.items():
            print(f"[bon-vs-pps] pps:{label}")
            for phase in ("callback", "decode", "reward"):
                wall_ms = payload[phase]["wall_ms_mean"]
                gpu_ms = payload[phase]["gpu_ms_mean"]
                print(f"  {phase}: wall_ms={wall_ms:.3f} gpu_ms={gpu_ms:.3f} wall_minus_gpu_ms={wall_ms-gpu_ms:.3f}")
    if pre_drain_summary:
        print("")
        print("[bon-vs-pps] pps pre-drain summary (ms means)")
        for label, payload in pre_drain_summary.items():
            print(
                f"[bon-vs-pps] pps:{label} pre_callback_drain_ms={payload['pre_callback_drain_ms_mean']:.3f} eval_other_ms={payload['eval_other_ms_mean']:.3f} eval_other_minus_pre_drain_ms={payload['eval_other_minus_pre_drain_ms_mean']:.3f}"
            )
    if allocator_delta_summary:
        print("")
        print("[bon-vs-pps] pps allocator delta summary")
        for label, payload in allocator_delta_summary.items():
            print(
                f"[bon-vs-pps] pps:{label} alloc_retries_delta={payload['alloc_retries_delta_mean']:.3f} allocated_peak_bytes_delta={payload['allocated_peak_bytes_delta_mean']:.1f} reserved_peak_bytes_delta={payload['reserved_peak_bytes_delta_mean']:.1f}"
            )
    print("")


if __name__ == "__main__":
    main()
