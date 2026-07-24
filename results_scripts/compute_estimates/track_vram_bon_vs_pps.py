#!/usr/bin/env python3
"""Track per-step VRAM usage for BoN vs PPS on a single diffusion backbone.

For each method (BoN, PPS) and each prompt we record, at every diffusion step:
  - step index (1-based)
  - wall time since pipeline start (s)
  - current allocated GPU memory (bytes)
  - intra-step peak allocated GPU memory (bytes)  -- max during this step,
    captured by resetting peak stats at the end of each step.

Aggregates across stats prompts (mean, std). Also captures the run-wide peak
allocated bytes per prompt and reports its mean step + mean t_s.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_TO_IMAGE_DIR = REPO_ROOT / "Fk-Diffusion-Steering" / "text_to_image"
FKD_DIR = TEXT_TO_IMAGE_DIR / "fkd_diffusers"
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (TEXT_TO_IMAGE_DIR, FKD_DIR, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fks_utils import do_eval  # noqa: E402
from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL  # noqa: E402
from fkd_diffusers.fkd_pipeline_sd import (  # noqa: E402
    latent_to_decode as latent_to_decode_sd,
)
from fkd_diffusers.fkd_pipeline_sdxl import (  # noqa: E402
    latent_to_decode as latent_to_decode_sdxl,
)

from compare_bon_vs_pps_wallclock import (  # noqa: E402
    apply_indices,
    build_pipeline,
    compute_checkpoint_steps,
    decode_latents_to_tensor_sd35,
    estimate_x0_from_ddim_transition,
    estimate_x0_from_sd35_transition,
    read_prompts,
    set_cudnn_benchmark,
    synchronize_if_needed,
)


def cuda_memory_now(device: str) -> Tuple[int, int]:
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return 0, 0
    return (
        int(torch.cuda.memory_allocated(device)),
        int(torch.cuda.max_memory_allocated(device)),
    )


def reset_peak(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def empty_cache(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def decode_pps_eval(
    *,
    pipeline,
    is_sd35: bool,
    latents: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    def decode_once(latent_batch: torch.Tensor) -> torch.Tensor:
        if is_sd35:
            return decode_latents_to_tensor_sd35(
                pipeline=pipeline, latents=latent_batch
            )
        decode_fn = (
            latent_to_decode_sdxl
            if isinstance(pipeline, FKDStableDiffusionXL)
            else latent_to_decode_sd
        )
        return decode_fn(model=pipeline, output_type="pil", latents=latent_batch)

    batch = int(latents.shape[0])
    if chunk_size <= 0 or chunk_size >= batch:
        return decode_once(latents)
    decoded_chunks = [decode_once(chunk) for chunk in latents.split(chunk_size)]
    return torch.cat(decoded_chunks, dim=0)


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
    pps_offload_first_n_evals: int,
    pps_vae_decode_batch: int,
) -> Dict:
    if method == "bon":
        init_particles = 4
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

    eval_step_set = set(eval_steps)
    callback_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    if is_sd35:
        callback_inputs.extend(
            ["pooled_prompt_embeds", "negative_pooled_prompt_embeds"]
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

    samples: List[Dict[str, float]] = []
    offload_count = 0
    offload_wall_s = 0.0
    backbone_module = pipeline.transformer if is_sd35 else pipeline.unet
    offload_eval_step_set = set()
    if method == "pps" and pps_offload_first_n_evals > 0:
        offload_eval_step_set = set(sorted(eval_steps)[:pps_offload_first_n_evals])

    def callback_on_step_end(_pipeline, step_idx, t, callback_kwargs):
        nonlocal prev_latents, prompt_list, offload_count, offload_wall_s
        latents_after_step = callback_kwargs.get("latents", prev_latents)
        step_number = step_idx + 1
        outputs = {"latents": latents_after_step}

        if step_number in eval_step_set:
            did_offload = False
            if step_number in offload_eval_step_set:
                synchronize_if_needed(device)
                offload_start = time.perf_counter()
                backbone_module.to("cpu")
                empty_cache(device)
                synchronize_if_needed(device)
                offload_wall_s += time.perf_counter() - offload_start
                did_offload = True
            decode_chunk_size = int(pps_vae_decode_batch) if method == "pps" else 0
            if is_sd35:
                x0_preds = estimate_x0_from_sd35_transition(
                    scheduler=pipeline.scheduler,
                    timestep=t,
                    sample_before_step=prev_latents,
                    sample_after_step=latents_after_step,
                )
                image_tensor = decode_pps_eval(
                    pipeline=pipeline,
                    is_sd35=is_sd35,
                    latents=x0_preds,
                    chunk_size=decode_chunk_size,
                ).detach()
            else:
                x0_preds = estimate_x0_from_ddim_transition(
                    scheduler=pipeline.scheduler,
                    timestep=t,
                    sample_before_step=prev_latents,
                    sample_after_step=latents_after_step,
                    num_inference_steps=time_steps,
                )
                image_tensor = decode_pps_eval(
                    pipeline=pipeline,
                    is_sd35=is_sd35,
                    latents=x0_preds,
                    chunk_size=decode_chunk_size,
                ).detach()

            eval_out = do_eval(
                prompt=prompt_list,
                images=image_tensor,
                metrics_to_compute=["ImageReward"],
            )
            rewards = torch.tensor(
                eval_out["ImageReward"]["result"],
                device=latents_after_step.device,
                dtype=torch.float32,
            )
            del image_tensor

            batch_before = int(latents_after_step.shape[0])
            keep_count = int(keep_by_eval_step.get(step_number, batch_before))
            keep_count = max(1, min(batch_before, keep_count))
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
            if did_offload:
                reload_start = time.perf_counter()
                backbone_module.to(device)
                synchronize_if_needed(device)
                offload_wall_s += time.perf_counter() - reload_start
                offload_count += 1

        synchronize_if_needed(device)
        t_now = time.perf_counter() - run_start
        cur_alloc, peak_alloc = cuda_memory_now(device)
        samples.append(
            {
                "step": int(step_number),
                "t_s": float(t_now),
                "alloc_bytes": int(cur_alloc),
                "peak_alloc_bytes": int(peak_alloc),
                "is_eval": bool(step_number in eval_step_set),
            }
        )
        reset_peak(device)
        prev_latents = outputs["latents"]
        return outputs

    empty_cache(device)
    reset_peak(device)
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

    if samples:
        peak_idx = max(
            range(len(samples)), key=lambda i: samples[i]["peak_alloc_bytes"]
        )
        run_peak_bytes = int(samples[peak_idx]["peak_alloc_bytes"])
        run_peak_step = int(samples[peak_idx]["step"])
        run_peak_t_s = float(samples[peak_idx]["t_s"])
    else:
        run_peak_bytes = 0
        run_peak_step = 0
        run_peak_t_s = 0.0

    return {
        "prompt_id": int(prompt_id),
        "method": method,
        "init_particles": int(init_particles),
        "total_wall_s": float(total_s),
        "run_peak_alloc_bytes": run_peak_bytes,
        "run_peak_step": run_peak_step,
        "run_peak_t_s": run_peak_t_s,
        "offload_count": int(offload_count),
        "offload_wall_s": float(offload_wall_s),
        "trace": samples,
    }


def aggregate_traces(prompt_traces: List[Dict]) -> List[Dict[str, float]]:
    """Average per-step values across stats prompts."""
    if not prompt_traces:
        return []
    n_steps = len(prompt_traces[0]["trace"])
    for tr in prompt_traces:
        if len(tr["trace"]) != n_steps:
            raise ValueError(
                f"Inconsistent trace length: prompt {tr['prompt_id']} has "
                f"{len(tr['trace'])} samples, expected {n_steps}"
            )
    averaged: List[Dict[str, float]] = []
    for i in range(n_steps):
        ts = [tr["trace"][i]["t_s"] for tr in prompt_traces]
        cur = [tr["trace"][i]["alloc_bytes"] for tr in prompt_traces]
        peak = [tr["trace"][i]["peak_alloc_bytes"] for tr in prompt_traces]
        step = int(prompt_traces[0]["trace"][i]["step"])
        is_eval = bool(prompt_traces[0]["trace"][i]["is_eval"])
        m_t, s_t = mean_std(ts)
        m_c, s_c = mean_std(cur)
        m_p, s_p = mean_std(peak)
        averaged.append(
            {
                "step": step,
                "is_eval": is_eval,
                "mean_t_s": m_t,
                "std_t_s": s_t,
                "mean_alloc_bytes": m_c,
                "std_alloc_bytes": s_c,
                "mean_peak_alloc_bytes": m_p,
                "std_peak_alloc_bytes": s_p,
                "n": len(prompt_traces),
            }
        )
    return averaged


def aggregate_peaks(prompt_traces: List[Dict]) -> Dict[str, float]:
    if not prompt_traces:
        return {
            "mean_peak_alloc_bytes": 0.0,
            "std_peak_alloc_bytes": 0.0,
            "mean_peak_step": 0.0,
            "std_peak_step": 0.0,
            "mean_peak_t_s": 0.0,
            "std_peak_t_s": 0.0,
            "mean_offload_count": 0.0,
            "std_offload_count": 0.0,
            "mean_offload_wall_s": 0.0,
            "std_offload_wall_s": 0.0,
            "n": 0,
        }
    peaks = [tr["run_peak_alloc_bytes"] for tr in prompt_traces]
    steps = [tr["run_peak_step"] for tr in prompt_traces]
    ts = [tr["run_peak_t_s"] for tr in prompt_traces]
    offload_counts = [tr.get("offload_count", 0) for tr in prompt_traces]
    offload_walls = [tr.get("offload_wall_s", 0.0) for tr in prompt_traces]
    m_p, s_p = mean_std(peaks)
    m_s, s_s = mean_std(steps)
    m_t, s_t = mean_std(ts)
    m_oc, s_oc = mean_std(offload_counts)
    m_ow, s_ow = mean_std(offload_walls)
    return {
        "mean_peak_alloc_bytes": m_p,
        "std_peak_alloc_bytes": s_p,
        "mean_peak_step": m_s,
        "std_peak_step": s_s,
        "mean_peak_t_s": m_t,
        "std_peak_t_s": s_t,
        "mean_offload_count": m_oc,
        "std_offload_count": s_oc,
        "mean_offload_wall_s": m_ow,
        "std_offload_wall_s": s_ow,
        "n": len(prompt_traces),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track per-step VRAM usage for BoN vs PPS."
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
        "--pps-offload-first-n-evals",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="For PPS, offload backbone before first N eval checkpoints.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
    )
    parser.add_argument(
        "--pps-vae-decode-batch",
        type=int,
        default=0,
        help="Chunk size for PPS VAE decode at eval steps (0 means full batch decode).",
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
    if args.pps_vae_decode_batch < 0:
        raise ValueError("--pps-vae-decode-batch must be >= 0")
    requested_methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    if not requested_methods:
        raise ValueError("--methods must include at least one method")
    valid_methods = {"bon", "pps"}
    unknown_methods = [m for m in requested_methods if m not in valid_methods]
    if unknown_methods:
        raise ValueError(f"Unknown methods in --methods: {unknown_methods}")
    selected_methods = list(dict.fromkeys(requested_methods))
    if args.pps_offload_first_n_evals > 0 and "pps" not in selected_methods:
        raise ValueError("--pps-offload-first-n-evals requires methods to include pps")
    if args.pps_vae_decode_batch > 0 and "pps" not in selected_methods:
        raise ValueError("--pps-vae-decode-batch requires methods to include pps")
    cudnn_state = set_cudnn_benchmark(args.cudnn_benchmark)

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

    pps_eval_steps = compute_checkpoint_steps(args.time_steps, [0.25, 0.5, 1.0])
    bon_eval_steps = [args.time_steps]
    pps_keep = {
        pps_eval_steps[0]: 4,
        pps_eval_steps[1]: 2,
        pps_eval_steps[2]: 2,
    }
    bon_keep = {args.time_steps: 4}

    args_payload = vars(args).copy()
    args_payload["method_configs"] = {
        "bon": {"init_particles": 4, "eval_steps": bon_eval_steps},
        "pps": {
            "init_particles": 8,
            "eval_steps": pps_eval_steps,
            "keep_by_step": pps_keep,
        },
    }
    args_payload["selected_methods"] = selected_methods
    args_payload["cudnn_benchmark_state"] = cudnn_state
    with open(output_dir / "args.json", "w", encoding="utf-8") as h:
        json.dump(args_payload, h, indent=2)

    pipeline, is_sd35 = build_pipeline(args.model_name, args.device)
    pipeline.set_progress_bar_config(disable=True)

    dropped_prompt_ids = [
        args.prompt_start_id + i
        for i in range(args.num_prompts_total - args.num_prompts_stats)
    ]
    stats_prompt_ids = set(
        args.prompt_start_id + i
        for i in range(args.num_prompts_total - args.num_prompts_stats, args.num_prompts_total)
    )

    per_prompt: Dict[str, List[Dict]] = {method: [] for method in selected_methods}
    total_items = len(selected_prompts) * len(selected_methods)
    with tqdm(total=total_items, desc=f"vram[{args.model_name}]", unit="run") as pbar:
        for local_idx, prompt in enumerate(selected_prompts):
            prompt_id = args.prompt_start_id + local_idx
            seed_base = args.seed + prompt_id * 100
            for method in selected_methods:
                if method == "bon":
                    eval_steps, keep = bon_eval_steps, bon_keep
                else:
                    eval_steps, keep = pps_eval_steps, pps_keep
                rec = run_single_method(
                    pipeline=pipeline,
                    is_sd35=is_sd35,
                    prompt=prompt,
                    method=method,
                    prompt_id=prompt_id,
                    seed_base=seed_base,
                    device=args.device,
                    time_steps=args.time_steps,
                    eta=args.eta,
                    eval_steps=eval_steps,
                    keep_by_eval_step=keep,
                    pps_offload_first_n_evals=args.pps_offload_first_n_evals,
                    pps_vae_decode_batch=args.pps_vae_decode_batch,
                )
                per_prompt[method].append(rec)
                pbar.update(1)

    # Filter to stats prompts for aggregation; keep warmup traces for inspection.
    stats_per_prompt: Dict[str, List[Dict]] = {
        method: [r for r in per_prompt[method] if r["prompt_id"] in stats_prompt_ids]
        for method in selected_methods
    }

    averaged: Dict[str, List[Dict[str, float]]] = {
        method: aggregate_traces(stats_per_prompt[method])
        for method in selected_methods
    }
    peak_summary: Dict[str, Dict[str, float]] = {
        method: aggregate_peaks(stats_per_prompt[method])
        for method in selected_methods
    }

    payload = {
        "config": {
            "model_name": args.model_name,
            "device": args.device,
            "time_steps": args.time_steps,
            "seed": args.seed,
            "prompt_start_id": args.prompt_start_id,
            "num_prompts_total": args.num_prompts_total,
            "num_prompts_stats": args.num_prompts_stats,
            "dropped_warmup_prompt_ids": dropped_prompt_ids,
            "stats_prompt_ids": sorted(stats_prompt_ids),
            "cudnn_benchmark": args.cudnn_benchmark,
            "cudnn_benchmark_state": cudnn_state,
            "selected_methods": selected_methods,
            "pps_offload_first_n_evals": args.pps_offload_first_n_evals,
            "pps_vae_decode_batch": args.pps_vae_decode_batch,
            "method_configs": args_payload["method_configs"],
        },
        "averaged": averaged,
        "peak_summary": peak_summary,
        "per_prompt": per_prompt,
    }

    with open(output_dir / "vram_trace.json", "w", encoding="utf-8") as h:
        json.dump(payload, h, indent=2)

    print("")
    print(f"[vram] model={args.model_name} time_steps={args.time_steps}")
    print(f"[vram] cudnn_benchmark={args.cudnn_benchmark} state={cudnn_state}")
    print(f"[vram] selected_methods={selected_methods}")
    print(f"[vram] pps_offload_first_n_evals={args.pps_offload_first_n_evals}")
    print(f"[vram] pps_vae_decode_batch={args.pps_vae_decode_batch}")
    print(f"[vram] dropped warmup prompt_ids={dropped_prompt_ids}")
    print(f"[vram] stats prompt_ids={sorted(stats_prompt_ids)}")
    for method in selected_methods:
        s = peak_summary[method]
        gb = s["mean_peak_alloc_bytes"] / (1024 ** 3)
        gb_std = s["std_peak_alloc_bytes"] / (1024 ** 3)
        print(
            f"[vram] {method}: peak={gb:.3f} +/- {gb_std:.3f} GB  "
            f"step={s['mean_peak_step']:.1f} +/- {s['std_peak_step']:.1f}  "
            f"t_s={s['mean_peak_t_s']:.3f} +/- {s['std_peak_t_s']:.3f}  "
            f"offload_count={s['mean_offload_count']:.2f} +/- {s['std_offload_count']:.2f}  "
            f"offload_wall_s={s['mean_offload_wall_s']:.3f} +/- {s['std_offload_wall_s']:.3f}  "
            f"(n={s['n']})"
        )
    print(f"[vram] output: {output_dir}")


if __name__ == "__main__":
    main()
