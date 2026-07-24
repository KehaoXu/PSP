#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    dsearch_dir = repo_root / "DSearch"
    if str(dsearch_dir) not in sys.path:
        sys.path.insert(0, str(dsearch_dir))
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

from diffusers_patch.ddim_with_kl import ddim_step_KL  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DSearch nonp-matching search on Geneval prompts with FK-style outputs. "
            "SD3.5 uses stochastic scheduler step in place of DDIM transition."
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
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--guidance-reward-fn", type=str, choices=["ImageReward", "HumanPreference"], required=True)
    parser.add_argument("--metrics-to-compute", type=str, default="ImageReward#HumanPreference")

    # Keep names/roles aligned with DSearch inference_decoding_nonp.py.
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--oversamplerate", type=int, default=2)
    parser.add_argument("--w", type=float, default=2.0)
    parser.add_argument("--search_schudule", type=str, default="all")
    parser.add_argument("--drop_schudule", type=str, default="exponential")
    parser.add_argument("--replacerate", type=float, default=0.0)
    parser.add_argument("--duplicate_size", type=int, default=1)
    parser.add_argument("--variant", type=str, default="PM")

    parser.add_argument("--stochastic-sampling", action="store_true")
    parser.add_argument("--use-step-wrapper-stochastic", action="store_true")
    parser.add_argument("--gamma-target", type=float, default=None)
    parser.add_argument("--show-step-progress", action="store_true")
    parser.add_argument("--profile-time-breakdown", action="store_true")
    parser.add_argument("--effective-c-report-path", type=str, default="")
    parser.add_argument("--enable-attention-slicing", action="store_true")
    parser.add_argument("--enable-vae-slicing", action="store_true")
    parser.add_argument("--resume-skip-existing", action="store_true")
    return parser.parse_args()


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


def _search_probability(schedule: str, step_idx: int, total_steps: int) -> float:
    if schedule == "all":
        return 1.1
    if schedule == "uniform":
        return 0.7
    if schedule == "linear":
        return (step_idx + 10) / float(max(1, total_steps - 1))
    if schedule == "exponential":
        return float(1 - np.exp(-step_idx / (float(max(1, total_steps - 1)) / 3.0)))
    raise ValueError(f"Invalid search_schudule: {schedule}")


def _num_samples_schedule(drop_schedule: str, eval_sp_size: int, batch_size_per_gpu: int, total_steps: int) -> List[int]:
    if total_steps <= 1:
        return []
    n = total_steps - 1
    if drop_schedule == "exponential":
        return [
            max(eval_sp_size, int(batch_size_per_gpu * (eval_sp_size / batch_size_per_gpu) ** (t / n)))
            for t in range(n)
        ]
    if drop_schedule == "quadratic":
        return [
            max(eval_sp_size, int(eval_sp_size + (batch_size_per_gpu - eval_sp_size) * (1 - t / n) ** 2))
            for t in range(n)
        ]
    if drop_schedule == "linear":
        return [
            max(eval_sp_size, int(batch_size_per_gpu - t * (batch_size_per_gpu - eval_sp_size) / n))
            for t in range(n)
        ]
    if drop_schedule == "sigmoid":
        k = 10.0
        vals = [max(eval_sp_size, int(batch_size_per_gpu / (1 + np.exp(-k * (t / n - 0.5))))) for t in range(n)]
        vals.reverse()
        return vals
    return [eval_sp_size] * n


def _expected_a_over_t(schedule: str, total_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    vals: List[float] = []
    for i in range(total_steps - 1):
        vals.append(float(max(0.0, min(1.0, _search_probability(schedule, i, total_steps)))))
    return float(np.mean(vals)) if vals else 1.0


def _print_effective_c_report(args: argparse.Namespace) -> str:
    a_over_t = _expected_a_over_t(args.search_schudule, int(args.num_inference_steps))
    raw_c = float(args.w) * float(args.oversamplerate)
    expected_c_bar = 1.0 + a_over_t * (raw_c - 1.0)
    report = (
        f"Effective-C mapping\n"
        f"- paper tree width w(t) <- --w ({args.w})\n"
        f"- paper width multiplier <- --oversamplerate ({args.oversamplerate})\n"
        f"- search set ratio |A|/T <- E[p_search] ({a_over_t:.6f}) from --search_schudule={args.search_schudule}\n"
        f"- raw C (search steps) = w * oversamplerate = {args.w} * {args.oversamplerate} = {raw_c:.6f}\n"
        f"- expected C_bar = 1 + (|A|/T) * (rawC - 1) = 1 + {a_over_t:.6f} * ({raw_c:.6f} - 1) = {expected_c_bar:.6f}\n"
    )
    if args.model_key == "sd35":
        report += (
            "- sd35 note: DDIM transition is not valid for SD3.5 scheduler; using scheduler.step stochastic transition\n"
            "- sd35 note: DDIM KL term is unavailable under this scheduler; KL is logged as zero-proxy\n"
        )
    print(report)
    return report


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


def _subset_cfg_tensor(t: torch.Tensor, keep_idx: torch.Tensor) -> torch.Tensor:
    old_bs = t.shape[0] // 2
    if old_bs == 0:
        return t
    uncond = t[:old_bs][keep_idx]
    cond = t[old_bs:][keep_idx]
    return torch.cat([uncond, cond], dim=0)


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


def install_sd35_step_wrapper_stochastic(scheduler, gamma_target: float) -> None:
    original_step = scheduler.step
    gamma_cap = np.sqrt(2.0) - 1.0
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
        noise_std = sigma_f * np.sqrt(max((1.0 + gamma_eff) ** 2 - 1.0, 0.0))
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
    """
    FlowMatch scheduler keeps internal step indices. DSearch evaluates multiple
    duplicate transitions per logical timestep, so we must avoid mutating this
    internal cursor across duplicate calls.
    """
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


def _resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    preset = MODEL_PRESETS[args.model_key]
    if not args.model_name:
        args.model_name = preset.model_name
    if args.num_inference_steps is None:
        args.num_inference_steps = preset.default_steps
    if not args.prompts_path:
        args.prompts_path = str(
            _repo_root() / "Fk-Diffusion-Steering" / "text_to_image" / "prompt_files" / "geneval_metadata.jsonl"
        )
    if not args.output_root:
        args.output_root = str(_repo_root() / "results" / "dsearch_v2")
    if args.model_key == "sd35":
        if not args.stochastic_sampling:
            args.stochastic_sampling = True
        if not args.use_step_wrapper_stochastic:
            args.use_step_wrapper_stochastic = True
        if args.gamma_target is None:
            args.gamma_target = 0.005
    return args


def _save_winner_files(exp_root: Path, seed: int, rows: List[dict], shard_tag: str) -> None:
    csv_path = exp_root / f"dsearch_winner_geneval_seed{seed}_{shard_tag}.csv"
    jsonl_path = exp_root / f"dsearch_winner_geneval_seed{seed}_{shard_tag}.jsonl"
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _load_existing_winner_rows(exp_root: Path, seed: int, shard_tag: str) -> Dict[int, dict]:
    csv_path = exp_root / f"dsearch_winner_geneval_seed{seed}_{shard_tag}.csv"
    if not csv_path.exists():
        return {}
    loaded: Dict[int, dict] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                prompt_id = int(row.get("prompt_id", "-1"))
            except ValueError:
                continue
            if prompt_id >= 0:
                loaded[prompt_id] = row
    return loaded


def main() -> None:
    args = _resolve_defaults(parse_args())
    torch.set_grad_enabled(False)
    if args.variant != "PM":
        raise ValueError("This runner supports variant=PM only.")
    if args.bs != args.num_images:
        raise ValueError("Expected --bs == --num_images")
    if args.oversamplerate < 1:
        raise ValueError("--oversamplerate must be >= 1")

    report = _print_effective_c_report(args)
    if args.effective_c_report_path:
        report_path = Path(args.effective_c_report_path).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().isoformat()}] model={args.model_key} guidance={args.guidance_reward_fn}\n")
            f.write(report)

    final_metrics_to_compute = [m for m in args.metrics_to_compute.split("#") if m]
    if not final_metrics_to_compute:
        final_metrics_to_compute = [args.guidance_reward_fn]
    intermediate_metrics_to_compute = [args.guidance_reward_fn]

    prompt_rows = _read_geneval_rows(Path(args.prompts_path).resolve())
    if args.prompt_start_id < 0 or args.prompt_start_id >= len(prompt_rows):
        raise ValueError("--prompt-start-id out of range")
    remaining = len(prompt_rows) - args.prompt_start_id
    requested = remaining if args.num_prompts is None else args.num_prompts
    selected_count = min(remaining, requested)
    guidance_tag = "ir" if args.guidance_reward_fn == "ImageReward" else "hps"
    exp_name = f"dsearch_{args.model_key}_k4_t{args.num_inference_steps}_{guidance_tag}_geneval"
    exp_root = Path(args.output_root).resolve() / MODEL_PRESETS[args.model_key].model_tag / exp_name
    exp_root.mkdir(parents=True, exist_ok=True)
    shard_end = args.prompt_start_id + max(0, selected_count - 1)
    shard_tag = f"p{args.prompt_start_id:05d}-{shard_end:05d}"

    prior_seed_dirs = sorted(
        [p for p in exp_root.glob(f"seed={args.seed}_*") if p.is_dir()]
    )
    selected_pairs: List[Tuple[int, dict]] = []
    skipped_prompt_ids: List[int] = []
    for prompt_id in range(args.prompt_start_id, args.prompt_start_id + selected_count):
        if args.resume_skip_existing:
            done = False
            for prev_dir in prior_seed_dirs:
                prev_prompt = prev_dir / f"{prompt_id:05d}"
                if (prev_prompt / "results.json").exists() and (prev_prompt / "best_of_n_samples" / "00000.png").exists():
                    done = True
                    break
            if done:
                skipped_prompt_ids.append(prompt_id)
                continue
        selected_pairs.append((prompt_id, prompt_rows[prompt_id]))

    seed_dir = exp_root / f"seed={args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    seed_dir.mkdir(parents=True, exist_ok=False)

    args_to_save = vars(args).copy()
    args_to_save["resolved_num_prompts"] = len(selected_pairs)
    args_to_save["requested_num_prompts"] = selected_count
    args_to_save["skipped_existing_prompt_ids"] = skipped_prompt_ids
    (seed_dir / "args.json").write_text(json.dumps(args_to_save, indent=2), encoding="utf-8")

    pipeline, is_sd35 = build_pipeline(args.model_name, args.device)
    pipeline.set_progress_bar_config(disable=True)
    if args.enable_attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    if args.enable_vae_slicing and hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
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

    agg_metrics: Dict[str, Dict[str, float]] = {m: {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0} for m in final_metrics_to_compute}
    winner_rows: List[dict] = []
    audit_rows: List[dict] = []
    total_particle_backbone_calls = 0
    total_prompt_time = 0.0
    total_pipeline_wall_time = 0.0
    total_reward_eval_time = 0.0
    total_reward_eval_calls = 0
    n_prompts = 0

    for prompt_id, row in tqdm(selected_pairs, desc="Prompts", unit="prompt"):
        prompt_text = str(row.get("prompt", "")).strip()
        if not prompt_text:
            continue

        prompt_dir = seed_dir / f"{prompt_id:05d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "metadata.jsonl").write_text(json.dumps(row), encoding="utf-8")

        eval_sp_size = int(args.bs)
        pop_size = eval_sp_size * int(args.oversamplerate) if args.drop_schudule is not None else eval_sp_size
        prompt_list = [prompt_text] * pop_size
        gens = [torch.Generator(device=args.device).manual_seed(args.seed + prompt_id * pop_size + i) for i in range(pop_size)]

        timesteps = pipeline.scheduler.timesteps
        num_samples_schedule = _num_samples_schedule(args.drop_schudule, eval_sp_size, eval_sp_size * int(args.oversamplerate), len(timesteps))

        if is_sd35:
            guidance_scale = 7.0
            pipeline._guidance_scale = guidance_scale
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
                    batch_size=pop_size,
                    num_channels_latents=pipeline.transformer.config.in_channels,
                    height=pipeline.transformer.config.sample_size * pipeline.vae_scale_factor,
                    width=pipeline.transformer.config.sample_size * pipeline.vae_scale_factor,
                    dtype=pipeline.transformer.dtype,
                    device=pipeline.device,
                    generator=list(gens),
                    latents=None,
                )
            add_text_embeds = None
            add_time_ids = None
        else:
            guidance_scale = 7.5
            pipeline._guidance_scale = guidance_scale
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
                text_encoder_projection_dim = (
                    int(ppe.shape[-1]) if pipeline.text_encoder_2 is None else pipeline.text_encoder_2.config.projection_dim
                )
                h = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
                w = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
                add_t = pipeline._get_add_time_ids(
                    (h, w), (0, 0), (h, w), dtype=prompt_embeds.dtype, text_encoder_projection_dim=text_encoder_projection_dim
                )
                neg_add_t = add_t
                add_time_ids = torch.cat([neg_add_t, add_t], dim=0).to(pipeline.device).repeat(pop_size, 1)
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
                    batch_size=pop_size,
                    num_channels_latents=pipeline.unet.config.in_channels,
                    height=pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                    width=pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                    dtype=pipeline.unet.dtype,
                    device=pipeline.device,
                    generator=gens,
                    latents=None,
                )

        prompt_particle_backbone_calls = 0
        search_events = 0
        prompt_reward_eval_time = 0.0
        prompt_reward_eval_calls = 0
        prompt_start = time.perf_counter()
        pipeline_start = time.perf_counter()
        if args.profile_time_breakdown and torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device=pipeline.device)
        step_pbar = tqdm(total=len(timesteps), desc=f"Prompt {prompt_id:05d} steps", unit="step", leave=False) if args.show_step_progress else None
        kl_proxy_sum = 0.0

        for i, t in enumerate(timesteps):
            cur_bs = latents.shape[0]
            if is_sd35:
                old_noise_pred = _predict_noise_sd35(pipeline, latents, t, prompt_embeds, pooled_prompt_embeds)
            else:
                old_noise_pred = _predict_noise_non_sd35(pipeline, latents, t, prompt_embeds, add_text_embeds, add_time_ids)
            prompt_particle_backbone_calls += int(cur_bs)

            if i < len(timesteps) - 1:
                drop_to = num_samples_schedule[i]
                p = _search_probability(args.search_schudule, i, len(timesteps))
                do_search = np.random.rand() < p
                w_ad = (eval_sp_size * int(args.oversamplerate) / cur_bs) if (eval_sp_size * int(args.oversamplerate) > cur_bs) else 1.0
                cur_duplicate = int(float(args.w) * int(args.duplicate_size) * w_ad) if do_search else int(args.duplicate_size)
                cur_duplicate = max(1, cur_duplicate)
                if do_search:
                    search_events += 1

                weights_list: List[torch.Tensor] = []
                latents_list: List[torch.Tensor] = []
                t_next = timesteps[i + 1]

                for _ in range(cur_duplicate):
                    if is_sd35:
                        # SD3.5 workaround: no DDIM step; use scheduler stochastic step.
                        latents_duplicate = _sd35_step_stateless(pipeline.scheduler, old_noise_pred, t, latents)
                        kl_terms = torch.zeros(cur_bs, device=latents.device, dtype=latents.dtype)
                    else:
                        latents_duplicate, kl_terms = ddim_step_KL(
                            pipeline.scheduler,
                            old_noise_pred,
                            old_noise_pred,
                            t,
                            latents,
                            eta=args.eta,
                        )
                    kl_proxy_sum += float(kl_terms.mean().item())

                    if is_sd35:
                        # Score each duplicate exactly as the next-step callback would:
                        # treat candidate latents as prev_latents for timestep t_next.
                        cb_kwargs = {
                            "latents": latents_duplicate,
                            "prompt_embeds": prompt_embeds,
                            "pooled_prompt_embeds": pooled_prompt_embeds,
                        }
                        x0_preds = compute_x0_preds_sd35(
                            pipeline=pipeline,
                            prev_latents=latents_duplicate,
                            t=t_next,
                            callback_kwargs=cb_kwargs,
                        )
                        image_tensor = decode_latents_to_tensor_sd35(pipeline=pipeline, latents=x0_preds).detach()
                    else:
                        cb_kwargs = {"prompt_embeds": prompt_embeds}
                        if add_text_embeds is not None:
                            cb_kwargs["add_text_embeds"] = add_text_embeds
                        if add_time_ids is not None:
                            cb_kwargs["add_time_ids"] = add_time_ids
                        x0_preds = compute_x0_preds_non_sd35(
                            pipeline=pipeline,
                            scheduler=pipeline.scheduler,
                            prev_latents=latents_duplicate,
                            t=t_next,
                            callback_kwargs=cb_kwargs,
                            eta=args.eta,
                        )
                        image_tensor = _decode_non_sd35_x0(pipeline, x0_preds)

                    prompt_particle_backbone_calls += int(cur_bs)
                    images = pipeline.image_processor.postprocess(image_tensor, output_type="pil")
                    eval_start = time.perf_counter()
                    metric_res = do_eval(prompt=prompt_list, images=images, metrics_to_compute=intermediate_metrics_to_compute)
                    prompt_reward_eval_time += time.perf_counter() - eval_start
                    prompt_reward_eval_calls += 1

                    weights = torch.tensor(metric_res[args.guidance_reward_fn]["result"], device=latents.device, dtype=torch.float32)
                    weights_list.append(weights.cpu().detach())
                    latents_list.append(latents_duplicate.cpu().detach())
                    del image_tensor, images, metric_res, x0_preds, latents_duplicate

                weights_st = torch.stack(weights_list)
                latents_st = torch.stack(latents_list)
                index_chosen = torch.argmax(weights_st, dim=0)
                latents = torch.stack([latents_st[index_chosen[j], j] for j in range(cur_bs)]).to(pipeline.device)
                del weights_list, latents_list, latents_st, index_chosen

                if cur_bs > drop_to:
                    max_weights = torch.max(weights_st, dim=0).values
                    top_idx = torch.argsort(max_weights, descending=True)[:drop_to]
                    top_idx_dev = top_idx.to(pipeline.device)
                    latents = latents[top_idx_dev]
                    prompt_list = [prompt_list[int(k)] for k in top_idx.tolist()]
                    prompt_embeds = _subset_cfg_tensor(prompt_embeds, top_idx_dev)
                    if add_text_embeds is not None:
                        add_text_embeds = _subset_cfg_tensor(add_text_embeds, top_idx_dev)
                    if add_time_ids is not None:
                        add_time_ids = _subset_cfg_tensor(add_time_ids, top_idx_dev)
                    if is_sd35:
                        pooled_prompt_embeds = _subset_cfg_tensor(pooled_prompt_embeds, top_idx_dev)
                    del max_weights, top_idx, top_idx_dev
                del weights_st
            else:
                if is_sd35:
                    latents = _sd35_step_stateless(pipeline.scheduler, old_noise_pred, t, latents)
                else:
                    latents, kl_terms = ddim_step_KL(
                        pipeline.scheduler,
                        old_noise_pred,
                        old_noise_pred,
                        t,
                        latents,
                        eta=args.eta,
                    )
                    kl_proxy_sum += float(kl_terms.mean().item())

            if step_pbar is not None:
                step_pbar.update(1)

        if step_pbar is not None:
            step_pbar.close()
        prompt_pipeline_wall_time = time.perf_counter() - pipeline_start

        if is_sd35:
            final_tensor = decode_latents_to_tensor_sd35(pipeline=pipeline, latents=latents).detach()
        else:
            final_tensor = _decode_non_sd35_x0(pipeline, latents)
        final_images = pipeline.image_processor.postprocess(final_tensor, output_type="pil")
        del final_tensor

        eval_start = time.perf_counter()
        final_metric_res = do_eval(prompt=prompt_list, images=final_images, metrics_to_compute=final_metrics_to_compute)
        prompt_reward_eval_time += time.perf_counter() - eval_start
        prompt_reward_eval_calls += 1

        prompt_elapsed = time.perf_counter() - prompt_start
        prompt_diffusion_approx_time = max(0.0, prompt_pipeline_wall_time - prompt_reward_eval_time)
        total_prompt_time += prompt_elapsed
        total_pipeline_wall_time += prompt_pipeline_wall_time
        total_reward_eval_time += prompt_reward_eval_time
        total_reward_eval_calls += prompt_reward_eval_calls
        total_particle_backbone_calls += prompt_particle_backbone_calls
        n_prompts += 1

        guidance = np.asarray(final_metric_res[args.guidance_reward_fn]["result"], dtype=np.float64)
        order = np.argsort(guidance)[::-1]
        top = order[: min(eval_sp_size, len(order))]
        sorted_images = [final_images[i] for i in top]
        for metric in final_metrics_to_compute:
            metric_vals = np.asarray(final_metric_res[metric]["result"], dtype=np.float64)[top]
            final_metric_res[metric] = _metric_stats(metric_vals.tolist())
            agg_metrics[metric]["mean"] += final_metric_res[metric]["mean"]
            agg_metrics[metric]["max"] += final_metric_res[metric]["max"]
            agg_metrics[metric]["min"] += final_metric_res[metric]["min"]
            agg_metrics[metric]["std"] += final_metric_res[metric]["std"]

        final_metric_res["time_taken"] = float(prompt_elapsed)
        final_metric_res["search_events"] = int(search_events)
        final_metric_res["particle_backbone_calls"] = int(prompt_particle_backbone_calls)
        final_metric_res["kl_proxy_mean"] = float(kl_proxy_sum / max(1, len(timesteps)))
        if args.profile_time_breakdown:
            final_metric_res["profile_pipeline_wall_time"] = float(prompt_pipeline_wall_time)
            final_metric_res["profile_reward_eval_time"] = float(prompt_reward_eval_time)
            final_metric_res["profile_diffusion_approx_time"] = float(prompt_diffusion_approx_time)
            final_metric_res["profile_reward_eval_calls"] = int(prompt_reward_eval_calls)
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                final_metric_res["profile_peak_vram_allocated_gib"] = float(
                    torch.cuda.max_memory_allocated(device=pipeline.device) / (1024 ** 3)
                )
                final_metric_res["profile_peak_vram_reserved_gib"] = float(
                    torch.cuda.max_memory_reserved(device=pipeline.device) / (1024 ** 3)
                )
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

        winner_row = {
            "prompt_id": int(prompt_id),
            "prompt": prompt_text,
            "seed": int(args.seed),
            "guidance_reward_fn": args.guidance_reward_fn,
            "guidance_score": float(final_metric_res[args.guidance_reward_fn]["result"][0]),
            "sample_relpath": str(Path(seed_dir.name) / f"{prompt_id:05d}" / "best_of_n_samples" / "00000.png"),
        }
        if "ImageReward" in final_metric_res:
            winner_row["image_reward"] = float(final_metric_res["ImageReward"]["result"][0])
        if "HumanPreference" in final_metric_res:
            winner_row["human_preference"] = float(final_metric_res["HumanPreference"]["result"][0])
        winner_rows.append(winner_row)
        audit_rows.append(
            {
                "prompt_id": int(prompt_id),
                "prompt": prompt_text,
                "particle_backbone_calls": int(prompt_particle_backbone_calls),
                "search_events": int(search_events),
                "prompt_wall_time": float(prompt_elapsed),
                "pipeline_wall_time": float(prompt_pipeline_wall_time),
                "reward_eval_time": float(prompt_reward_eval_time),
                "kl_proxy_mean": float(kl_proxy_sum / max(1, len(timesteps))),
            }
        )

    existing_winner_rows = _load_existing_winner_rows(exp_root, args.seed, shard_tag) if args.resume_skip_existing else {}
    merged_winner_rows = existing_winner_rows.copy()
    for row in winner_rows:
        try:
            merged_winner_rows[int(row["prompt_id"])] = row
        except Exception:
            continue

    if n_prompts == 0:
        if args.resume_skip_existing:
            summary = {
                "resume_skip_existing": True,
                "requested_num_prompts": int(selected_count),
                "resolved_num_prompts": 0,
                "skipped_existing_prompt_ids": skipped_prompt_ids,
            }
            (seed_dir / "final_metrics.json").write_text(json.dumps(summary), encoding="utf-8")
            with (seed_dir / "compute_audit.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "prompt_id",
                        "prompt",
                        "particle_backbone_calls",
                        "search_events",
                        "prompt_wall_time",
                        "pipeline_wall_time",
                        "reward_eval_time",
                        "kl_proxy_mean",
                    ],
                )
                writer.writeheader()
            merged_rows = [merged_winner_rows[k] for k in sorted(merged_winner_rows.keys())]
            _save_winner_files(exp_root, args.seed, merged_rows, shard_tag)
            print(f"No pending prompts. Reused existing outputs for seed={args.seed}, shard={shard_tag}.")
            print(f"Done. Output seed dir: {seed_dir}")
            return
        raise RuntimeError("No prompts were processed.")

    for metric in final_metrics_to_compute:
        for key in ["mean", "max", "min", "std"]:
            agg_metrics[metric][key] /= float(n_prompts)

    empirical_effective_c = float(total_particle_backbone_calls) / float(max(1, n_prompts * args.num_inference_steps))
    agg_metrics["compute_audit"] = {
        "total_particle_backbone_calls": int(total_particle_backbone_calls),
        "avg_particle_backbone_calls_per_prompt": float(total_particle_backbone_calls / n_prompts),
        "avg_particle_backbone_calls_per_timestep": float(total_particle_backbone_calls / (n_prompts * args.num_inference_steps)),
        "empirical_effective_c": float(empirical_effective_c),
        "expected_effective_c_report": report,
    }
    if args.profile_time_breakdown:
        agg_metrics["timing_profile"] = {
            "total_prompt_wall_time": float(total_prompt_time),
            "total_pipeline_wall_time": float(total_pipeline_wall_time),
            "total_reward_eval_time": float(total_reward_eval_time),
            "total_diffusion_approx_time": float(max(0.0, total_pipeline_wall_time - total_reward_eval_time)),
            "reward_eval_calls": int(total_reward_eval_calls),
            "avg_prompt_wall_time": float(total_prompt_time / n_prompts),
            "avg_pipeline_wall_time": float(total_pipeline_wall_time / n_prompts),
            "avg_reward_eval_time": float(total_reward_eval_time / n_prompts),
            "avg_diffusion_approx_time": float(max(0.0, total_pipeline_wall_time - total_reward_eval_time) / n_prompts),
        }
    (seed_dir / "final_metrics.json").write_text(json.dumps(agg_metrics), encoding="utf-8")

    with (seed_dir / "compute_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_id",
                "prompt",
                "particle_backbone_calls",
                "search_events",
                "prompt_wall_time",
                "pipeline_wall_time",
                "reward_eval_time",
                "kl_proxy_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    merged_rows = [merged_winner_rows[k] for k in sorted(merged_winner_rows.keys())]
    _save_winner_files(exp_root, args.seed, merged_rows, shard_tag)
    print(f"Done. Output seed dir: {seed_dir}")


if __name__ == "__main__":
    main()
