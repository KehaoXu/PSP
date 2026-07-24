#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, List, Sequence

import torch
from diffusers import DDIMScheduler
from tqdm import tqdm

from collect_image_reward_signal import (
    build_pipeline,
    chunked,
    compute_x0_preds_non_sd35,
    compute_x0_preds_sd35,
    decode_latents_to_tensor_sd35,
)
from fkd_diffusers.fkd_pipeline_sd import latent_to_decode as latent_to_decode_sd
from fkd_diffusers.fkd_pipeline_sdxl import (
    FKDStableDiffusionXL,
    latent_to_decode as latent_to_decode_sdxl,
)
from fks_utils import do_eval


def _normalize_multiprompt_entries(
    records: Sequence[dict], *, expected_expansion_count: int
) -> List[dict]:
    normalized: List[dict] = []
    seen_prompt_ids = set()
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"Record #{idx} in multiprompt JSON must be an object.")
        if "prompt_id" not in rec:
            raise ValueError(f"Record #{idx} missing required key: prompt_id")
        if "expansions" not in rec:
            raise ValueError(f"Record #{idx} missing required key: expansions")

        prompt_id = rec["prompt_id"]
        if not isinstance(prompt_id, int) or prompt_id < 0:
            raise ValueError(
                f"Record #{idx} has invalid prompt_id={prompt_id!r}; expected non-negative integer."
            )
        if prompt_id in seen_prompt_ids:
            raise ValueError(f"Duplicate prompt_id found in multiprompt JSON: {prompt_id}")
        seen_prompt_ids.add(prompt_id)

        original_prompt = rec.get("original_prompt", rec.get("prompt"))
        if not isinstance(original_prompt, str) or not original_prompt.strip():
            raise ValueError(
                f"Record prompt_id={prompt_id} must include non-empty original_prompt."
            )
        original_prompt = original_prompt.strip()

        expansions_raw = rec["expansions"]
        if not isinstance(expansions_raw, list):
            raise ValueError(
                f"Record prompt_id={prompt_id} has invalid expansions type; expected list."
            )
        if len(expansions_raw) != expected_expansion_count:
            raise ValueError(
                f"Record prompt_id={prompt_id} must have exactly {expected_expansion_count} expansions "
                f"(found {len(expansions_raw)})."
            )

        expansion_by_id: Dict[int, str] = {}
        for eidx, item in enumerate(expansions_raw):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Record prompt_id={prompt_id} expansion #{eidx} must be an object."
                )
            if "prompt_expansion_id" not in item:
                raise ValueError(
                    f"Record prompt_id={prompt_id} expansion #{eidx} missing prompt_expansion_id."
                )
            if "prompt" not in item:
                raise ValueError(
                    f"Record prompt_id={prompt_id} expansion #{eidx} missing prompt."
                )
            expansion_id = item["prompt_expansion_id"]
            expansion_prompt = item["prompt"]
            if not isinstance(expansion_id, int):
                raise ValueError(
                    f"Record prompt_id={prompt_id} expansion #{eidx} has non-integer prompt_expansion_id."
                )
            if expansion_id < 0 or expansion_id >= expected_expansion_count:
                raise ValueError(
                    f"Record prompt_id={prompt_id} expansion id {expansion_id} out of valid range "
                    f"[0, {expected_expansion_count - 1}]."
                )
            if expansion_id in expansion_by_id:
                raise ValueError(
                    f"Record prompt_id={prompt_id} has duplicate prompt_expansion_id={expansion_id}."
                )
            if not isinstance(expansion_prompt, str) or not expansion_prompt.strip():
                raise ValueError(
                    f"Record prompt_id={prompt_id} expansion id {expansion_id} has empty prompt text."
                )
            expansion_by_id[expansion_id] = expansion_prompt.strip()

        missing_ids = [
            i for i in range(expected_expansion_count) if i not in expansion_by_id
        ]
        if missing_ids:
            raise ValueError(
                f"Record prompt_id={prompt_id} missing expansion ids: {missing_ids[:10]}"
            )

        normalized.append(
            {
                "prompt_id": prompt_id,
                "original_prompt": original_prompt,
                "expansions": [
                    {
                        "prompt_expansion_id": i,
                        "prompt": expansion_by_id[i],
                    }
                    for i in range(expected_expansion_count)
                ],
            }
        )
    normalized.sort(key=lambda x: x["prompt_id"])
    return normalized


def read_multiprompt_records(path: str, *, expected_expansion_count: int) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(
            "Multiprompt JSON must be a list of records with prompt_id/original_prompt/expansions."
        )
    return _normalize_multiprompt_entries(
        payload, expected_expansion_count=expected_expansion_count
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect per-step ImageReward/HPS(pred_x0) using prompt expansions per seed index. "
            "Each seed index maps to one prompt expansion for the same original prompt."
        )
    )
    parser.add_argument("--prompts-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="stable-diffusion-v1-5")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-start-id", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=24)
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

    all_records = read_multiprompt_records(
        args.prompts_path, expected_expansion_count=args.num_seeds
    )
    if not all_records:
        raise ValueError("No prompt records found in the provided multiprompt file.")
    if args.prompt_start_id >= len(all_records):
        raise ValueError(
            f"--prompt-start-id ({args.prompt_start_id}) out of range for dataset size {len(all_records)}"
        )

    remaining = len(all_records) - args.prompt_start_id
    selected_count = args.num_prompts if args.num_prompts is not None else remaining
    selected_count = min(selected_count, remaining)
    prompt_end_id = args.prompt_start_id + selected_count
    selected_records = all_records[args.prompt_start_id:prompt_end_id]

    first_prompt_id = selected_records[0]["prompt_id"]
    last_prompt_id = selected_records[-1]["prompt_id"]
    print(
        f"collecting multiprompt for entries {args.prompt_start_id} <= idx < {prompt_end_id} "
        f"(prompt_ids {first_prompt_id}..{last_prompt_id})"
    )

    os.makedirs(args.output_dir, exist_ok=False)
    metrics_csv_path = os.path.join(args.output_dir, "metrics.csv")
    samples_root = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_root, exist_ok=True)

    args_to_save = vars(args).copy()
    args_to_save["prompt_end_id"] = prompt_end_id
    args_to_save["resolved_num_prompts"] = selected_count
    args_to_save["prompt_id_start"] = first_prompt_id
    args_to_save["prompt_id_end"] = last_prompt_id
    print("Args:", json.dumps(args_to_save, indent=2))
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(args_to_save, f, indent=2)

    pipeline, is_sd35 = build_pipeline(args.model_name, args.device)
    pipeline.set_progress_bar_config(disable=True)
    normalized_model_name = args.model_name.strip().lower()
    is_dpo_model = "mhdang/dpo-" in normalized_model_name
    collect_hps = not args.no_hps
    metrics_to_compute = ["ImageReward"] + (["HumanPreference"] if collect_hps else [])

    total_images = len(selected_records) * args.num_seeds
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

        for local_prompt_idx, rec in enumerate(selected_records):
            _ = local_prompt_idx
            prompt_id = rec["prompt_id"]
            expansions = rec["expansions"]
            prompt_sample_dir = os.path.join(samples_root, f"{prompt_id:05d}")
            os.makedirs(prompt_sample_dir, exist_ok=True)

            # Keep identical seed->noise mapping across all prompt_ids. For a given prompt_id,
            # seed index i uses expansion i and generator seed (args.seed + i).
            prompt_seeds = [args.seed + i for i in range(args.num_seeds)]
            seed_indices = list(range(args.num_seeds))
            for batch_seed_indices in chunked(seed_indices, args.batch_size):
                batch = [prompt_seeds[i] for i in batch_seed_indices]
                batch_step_bar = tqdm(
                    total=args.time_steps,
                    desc=f"Prompt {prompt_id} batch",
                    unit="step",
                    leave=False,
                )
                prompt_list = [expansions[i]["prompt"] for i in batch_seed_indices]
                expansion_id_by_seed = {
                    seed: expansions[i]["prompt_expansion_id"]
                    for seed, i in zip(batch, batch_seed_indices)
                }
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
                        height=pipeline.transformer.config.sample_size
                        * pipeline.vae_scale_factor,
                        width=pipeline.transformer.config.sample_size
                        * pipeline.vae_scale_factor,
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
                        images = pipeline.image_processor.postprocess(
                            image_tensor, output_type="pil"
                        )
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
                    # Keep legacy callback scheduler behavior for non-DPO models.
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
                        images = pipeline.image_processor.postprocess(
                            image_tensor, output_type="pil"
                        )
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
                            hps_parts.append(
                                f"{label}%:{(running_pct_hps_sum[label] / cnt):.3f}"
                            )
                    if ir_parts:
                        postfix["IR@20/40/60/80"] = " ".join(ir_parts)
                    if collect_hps and hps_parts:
                        postfix["HPS@20/40/60/80"] = " ".join(hps_parts)
                    pbar.set_postfix(postfix)

                    point_labels = pct_labels + ["100"]
                    point_step_numbers = pct_step_numbers + [args.time_steps]
                    point_step_indices = pct_step_indices + [args.time_steps - 1]
                    snapshot_seed_count = min(4, len(batch))
                    snapshot_seeds = batch[:snapshot_seed_count]

                    pbar.write(
                        f"[batch sanity] prompt_id={prompt_id} seeds={snapshot_seeds} "
                        f"expansion_ids={[expansion_id_by_seed[s] for s in snapshot_seeds]} "
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
                                    hps_chunks.append(
                                        f"{label}%:{float(hps_vals[local_idx]):.4f}"
                                    )
                        line = (
                            f"  seed={seed} expansion_id={expansion_id_by_seed[seed]} "
                            f"IR[{', '.join(ir_chunks)}]"
                        )
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
                            batch_ir_parts.append(
                                f"{label}%:{(float(sum(ir_vals)) / len(ir_vals)):.4f}"
                            )
                        if collect_hps:
                            hps_vals = step_hps.get(idx_point)
                            if hps_vals is not None and len(hps_vals) > 0:
                                batch_hps_parts.append(
                                    f"{label}%:{(float(sum(hps_vals)) / len(hps_vals)):.4f}"
                                )

                        if label == "100":
                            if running_final_count > 0:
                                running_ir_parts.append(
                                    f"{label}%:{(running_final_ir_sum / running_final_count):.4f}"
                                )
                                if collect_hps:
                                    running_hps_parts.append(
                                        f"{label}%:{(running_final_hps_sum / running_final_count):.4f}"
                                    )
                        else:
                            cnt = running_pct_count[label]
                            if cnt > 0:
                                running_ir_parts.append(
                                    f"{label}%:{(running_pct_ir_sum[label] / cnt):.4f}"
                                )
                                if collect_hps:
                                    running_hps_parts.append(
                                        f"{label}%:{(running_pct_hps_sum[label] / cnt):.4f}"
                                    )

                    pbar.write(f"  batch_mean IR[{', '.join(batch_ir_parts)}]")
                    if collect_hps:
                        pbar.write(f"  batch_mean HPS[{', '.join(batch_hps_parts)}]")
                    pbar.write(
                        f"  running_mean IR[{', '.join(running_ir_parts)}] n={running_final_count}"
                    )
                    if collect_hps:
                        pbar.write(
                            f"  running_mean HPS[{', '.join(running_hps_parts)}] n={running_final_count}"
                        )

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
                    for seed, prompt_text, ir_score, hps_score in zip(
                        batch, prompt_list, image_rewards, hps_scores
                    ):
                        writer.writerow(
                            {
                                "prompt_id": prompt_id,
                                "prompt": prompt_text,
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
