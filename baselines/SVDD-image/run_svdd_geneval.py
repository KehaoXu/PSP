import argparse
import csv
import inspect
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image
from tqdm import tqdm

from imagereward_adapter import ImageRewardScorerAdapter
from sd_pipeline import Decoding_nonbatch_SDPipeline


BACKBONE_DEFAULTS = {
    "sd15": {
        "model_name": "runwayml/stable-diffusion-v1-5",
        "num_inference_steps": 64,
        "guidance_scale": 7.5,
        "height": 512,
        "width": 512,
        "torch_dtype": torch.float16,
    },
    "sdxl": {
        "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
        "num_inference_steps": 64,
        "guidance_scale": 7.5,
        "height": 1024,
        "width": 1024,
        "torch_dtype": torch.float16,
    },
    "sd35": {
        "model_name": "stabilityai/stable-diffusion-3.5-large",
        "num_inference_steps": 32,
        "guidance_scale": 7.0,
        "height": 1024,
        "width": 1024,
        "torch_dtype": torch.bfloat16,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="SVDD full Geneval runner")
    parser.add_argument("--model_key", type=str, required=True, choices=["sd15", "sdxl", "sd35"])
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--prompt_path", type=str, default="Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl")
    parser.add_argument("--output_root", type=str, default="results/svdd")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--prompt-start-id", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--reward", type=str, default="imagereward")
    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)

    parser.add_argument("--variant", type=str, default="PM")
    parser.add_argument("--duplicate_size", type=int, default=4)
    parser.add_argument("--eta", type=float, default=1.0)

    parser.add_argument("--stochastic_sampling", action="store_true")
    parser.add_argument("--use_step_wrapper_stochastic", action="store_true")
    parser.add_argument("--gamma_target", type=float, default=0.005)
    parser.add_argument("--resume-skip-existing", action="store_true")
    return parser.parse_args()


def load_geneval_metadata(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def select_prompts(all_prompts, prompt_start_id, num_prompts):
    if prompt_start_id < 0 or prompt_start_id >= len(all_prompts):
        raise ValueError("--prompt-start-id out of range")
    remaining = len(all_prompts) - prompt_start_id
    selected_count = remaining if num_prompts is None else min(num_prompts, remaining)
    selected = all_prompts[prompt_start_id : prompt_start_id + selected_count]
    if not selected:
        raise ValueError("No prompts selected")
    return selected, selected_count


def resolve_settings(args):
    defaults = BACKBONE_DEFAULTS[args.model_key]
    exp_name_default = {
        "sd15": "svdd_sd15_t64_ir_geneval",
        "sdxl": "svdd_sdxl_t64_ir_geneval",
        "sd35": "svdd_sd35_t32_ir_geneval",
    }[args.model_key]
    return {
        "model_name": args.model_name.strip() if args.model_name.strip() else defaults["model_name"],
        "num_inference_steps": args.num_inference_steps if args.num_inference_steps is not None else defaults["num_inference_steps"],
        "guidance_scale": args.guidance_scale if args.guidance_scale is not None else defaults["guidance_scale"],
        "height": args.height if args.height is not None else defaults["height"],
        "width": args.width if args.width is not None else defaults["width"],
        "torch_dtype": defaults["torch_dtype"],
        "exp_name": args.exp_name.strip() if args.exp_name.strip() else exp_name_default,
    }


def install_sd35_step_wrapper_stochastic(scheduler, gamma_target):
    gamma_cap = np.sqrt(2.0) - 1.0
    gamma_eff = float(np.clip(float(gamma_target), 0.0, gamma_cap))
    original_step = scheduler.step

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
        generator = step_kwargs.get("generator", None)
        if isinstance(generator, torch.Generator):
            noise = torch.randn(prev_sample.shape, generator=generator, device=prev_sample.device, dtype=prev_sample.dtype)
        else:
            noise = torch.randn_like(prev_sample)

        perturbed = prev_sample + noise * noise_std
        if isinstance(out, tuple):
            return (perturbed,) + out[1:]
        out.prev_sample = perturbed
        return out

    scheduler.step = wrapped_step


def decode_latents_to_tensor_sd35(*, pipeline, latents):
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
            device=decode_param.device,
            dtype=decode_param.dtype,
        )
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
            latents = latents + vae.config.shift_factor
        image = vae.decode(latents, return_dict=False)[0]

    if needs_upcast:
        vae.to(dtype=torch.float16)

    return image


def compute_x0_preds_sd35(*, pipeline, prev_latents, t, callback_kwargs):
    prompt_embeds = callback_kwargs["prompt_embeds"]
    pooled_prompt_embeds = callback_kwargs["pooled_prompt_embeds"]
    latent_model_input = torch.cat([prev_latents] * 2) if pipeline.do_classifier_free_guidance else prev_latents
    timestep = t.expand(latent_model_input.shape[0])

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
        noise_pred = noise_pred_uncond + pipeline.guidance_scale * (noise_pred_text - noise_pred_uncond)

    if hasattr(pipeline.scheduler, "index_for_timestep"):
        sigma_idx = pipeline.scheduler.index_for_timestep(t)
    else:
        timesteps = pipeline.scheduler.timesteps
        sigma_idx = int((timesteps == t).nonzero(as_tuple=True)[0][0].item())
    sigma = pipeline.scheduler.sigmas[sigma_idx].to(device=prev_latents.device, dtype=prev_latents.dtype)
    while sigma.ndim < prev_latents.ndim:
        sigma = sigma.view(*sigma.shape, 1)
    return prev_latents - sigma * noise_pred


def build_svdd_fkd_args(args, settings):
    return {
        "lmbda": 10.0,
        "num_particles": int(args.duplicate_size),
        "use_smc": True,
        "adaptive_resampling": False,
        "resample_frequency": 1,
        "time_steps": int(settings["num_inference_steps"]),
        "resampling_t_start": 0,
        "resampling_t_end": int(settings["num_inference_steps"]),
        "guidance_reward_fn": args.guidance_reward_fn,
        "potential_type": "diff",
        "resampling": "multinomial",
        "tempering_schedule": "constant",
        "log_reward_every_step": True,
    }


def run_sdxl_true_svdd(pipe, scorer, prompt, settings, args, seed, prompt_id, device):
    prompt_population = [prompt] * args.duplicate_size
    generators = [
        torch.Generator(device=device).manual_seed(seed + prompt_id * args.duplicate_size + idx)
        for idx in range(args.duplicate_size)
    ]
    fkd_args = build_svdd_fkd_args(args, settings)

    def custom_reward_fn(images, prompts):
        return scorer.score_pil(prompts, images)

    fkd_args["custom_reward_fn"] = custom_reward_fn

    output = pipe(
        prompt_population,
        num_inference_steps=settings["num_inference_steps"],
        guidance_scale=settings["guidance_scale"],
        height=settings["height"],
        width=settings["width"],
        num_images_per_prompt=1,
        generator=generators,
        output_type="pil",
        fkd_args=fkd_args,
    )
    images = output.images if hasattr(output, "images") else output[0]
    scores = scorer.score_pil(prompt_population, images)
    best_idx = int(np.argmax(np.array(scores)))
    return images[best_idx], float(scores[best_idx])


def run_sd35_true_svdd(pipe, scorer, prompt, settings, args, seed, prompt_id, device):
    from fkd_diffusers.fkd_class import FKD

    prompt_population = [prompt] * args.duplicate_size
    fkd_args = build_svdd_fkd_args(args, settings)
    generators = [
        torch.Generator(device=device).manual_seed(seed + prompt_id * args.duplicate_size + idx)
        for idx in range(args.duplicate_size)
    ]

    def reward_fn(decoded_tensor):
        pil_images = pipe.image_processor.postprocess(decoded_tensor, output_type="pil")
        rewards = scorer.score_pil(prompt_population, pil_images)
        # FKD resampling converts weights to numpy in smc_utils; keep reward dtype numpy-safe.
        return torch.tensor(rewards, device=decoded_tensor.device, dtype=torch.float32)

    fkd = FKD(
        latent_to_decode_fn=lambda x: decode_latents_to_tensor_sd35(pipeline=pipe, latents=x),
        reward_fn=reward_fn,
        **fkd_args,
    )

    callback_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "pooled_prompt_embeds",
        "negative_pooled_prompt_embeds",
    ]
    prev_latents = None

    def callback_on_step_end(_pipeline, step_idx, timestep_value, callback_kwargs):
        nonlocal prev_latents
        latents = callback_kwargs["latents"]
        if prev_latents is None:
            raise RuntimeError("prev_latents is not initialized for SD3.5 SVDD callback")
        x0_preds = compute_x0_preds_sd35(
            pipeline=pipe,
            prev_latents=prev_latents,
            t=timestep_value,
            callback_kwargs=callback_kwargs,
        )
        latents, _ = fkd.resample(sampling_idx=step_idx, latents=latents, x0_preds=x0_preds)
        prev_latents = latents
        return {"latents": latents}

    with torch.no_grad():
        init_latents = pipe.prepare_latents(
            batch_size=args.duplicate_size,
            num_channels_latents=pipe.transformer.config.in_channels,
            height=settings["height"],
            width=settings["width"],
            dtype=pipe.transformer.dtype,
            device=pipe.device,
            generator=list(generators),
            latents=None,
        )
        prev_latents = init_latents
        output = pipe(
            prompt_population,
            num_inference_steps=settings["num_inference_steps"],
            guidance_scale=settings["guidance_scale"],
            height=settings["height"],
            width=settings["width"],
            num_images_per_prompt=1,
            generator=list(generators),
            latents=init_latents,
            output_type="pil",
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_inputs,
        )

    images = output.images if hasattr(output, "images") else output[0]
    scores = scorer.score_pil(prompt_population, images)
    best_idx = int(np.argmax(np.array(scores)))
    return images[best_idx], float(scores[best_idx])


def build_pipeline(args, settings):
    repo_root = Path(__file__).resolve().parents[1]
    fk_text_to_image_dir = repo_root / "Fk-Diffusion-Steering" / "text_to_image"
    if str(fk_text_to_image_dir) not in sys.path:
        sys.path.append(str(fk_text_to_image_dir))
    fk_diffusers_dir = fk_text_to_image_dir / "fkd_diffusers"
    if str(fk_diffusers_dir) not in sys.path:
        sys.path.append(str(fk_diffusers_dir))

    if args.model_key == "sd15":
        pipe = Decoding_nonbatch_SDPipeline.from_pretrained(settings["model_name"], local_files_only=True)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        return pipe

    if args.model_key == "sdxl":
        from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL

        pipe = FKDStableDiffusionXL.from_pretrained(settings["model_name"], torch_dtype=settings["torch_dtype"])
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        if hasattr(pipe.vae, "config") and hasattr(pipe.vae.config, "force_upcast"):
            pipe.vae.config.force_upcast = True
        return pipe

    from fkd_diffusers.fkd_pipeline_sd3 import FKDStableDiffusion3

    pipe = FKDStableDiffusion3.from_pretrained(settings["model_name"], torch_dtype=settings["torch_dtype"])
    scheduler_cls = pipe.scheduler.__class__
    init_params = set(inspect.signature(scheduler_cls.__init__).parameters.keys())
    supports_stochastic = "stochastic_sampling" in init_params
    scheduler_stochastic = args.stochastic_sampling and not args.use_step_wrapper_stochastic
    if supports_stochastic:
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, stochastic_sampling=scheduler_stochastic)
    else:
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    if args.use_step_wrapper_stochastic:
        install_sd35_step_wrapper_stochastic(pipe.scheduler, args.gamma_target)
    return pipe


def save_prompt_outputs(prompt_dir, metadata_item, image, score):
    os.makedirs(prompt_dir, exist_ok=True)
    with open(os.path.join(prompt_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
        json.dump(metadata_item, f)

    samples_dir = os.path.join(prompt_dir, "samples")
    best_dir = os.path.join(prompt_dir, "best_of_n_samples")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    samples_path = os.path.join(samples_dir, "00000.png")
    best_path = os.path.join(best_dir, "00000.png")
    image.save(samples_path)
    image.save(best_path)

    score = float(score)
    metric_obj = {
        "result": [score],
        "mean": score,
        "std": 0.0,
        "max": score,
        "min": score,
    }
    with open(os.path.join(prompt_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"ImageReward": metric_obj}, f, indent=2)


def parse_seed_list(seed_text):
    seed_vals = [s.strip() for s in seed_text.split(",") if s.strip()]
    if not seed_vals:
        raise ValueError("No seeds provided")
    return [int(s) for s in seed_vals]


def write_seed_aggregates(seed_dir, winner_rows):
    winner_csv = os.path.join(seed_dir, "svdd_winner_rows.csv")
    with open(winner_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "prompt_id", "prompt", "image_reward"])
        writer.writeheader()
        writer.writerows(winner_rows)

    winner_jsonl = os.path.join(seed_dir, "svdd_winner_rows.jsonl")
    with open(winner_jsonl, "w", encoding="utf-8") as f:
        for row in winner_rows:
            f.write(json.dumps(row) + "\n")

    rewards = [float(r["image_reward"]) for r in winner_rows]
    final_metrics = {
        "num_prompts": len(winner_rows),
        "ImageReward": {
            "mean": float(np.mean(rewards)) if rewards else 0.0,
            "std": float(np.std(rewards)) if rewards else 0.0,
            "max": float(np.max(rewards)) if rewards else 0.0,
            "min": float(np.min(rewards)) if rewards else 0.0,
        },
    }
    with open(os.path.join(seed_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    audit_path = os.path.join(seed_dir, "compute_audit.csv")
    with open(audit_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "status"])
        writer.writeheader()
        for row in winner_rows:
            writer.writerow({"prompt_id": row["prompt_id"], "status": "ok"})


def main():
    args = parse_args()
    args.reward = args.reward.lower()
    settings = resolve_settings(args)

    if args.model_key == "sd35":
        args.stochastic_sampling = True
        args.use_step_wrapper_stochastic = True

    all_prompts = load_geneval_metadata(args.prompt_path)
    selected_prompts, selected_count = select_prompts(all_prompts, args.prompt_start_id, args.num_prompts)
    seeds = parse_seed_list(args.seeds)

    device = args.device
    pipe = build_pipeline(args, settings).to(device)
    scorer = ImageRewardScorerAdapter(device=device)

    if args.model_key == "sd15":
        pipe.setup_scorer(scorer)
        pipe.set_variant(args.variant)
        pipe.set_reward("imagereward")
        pipe.set_parameters(1, args.duplicate_size)

    exp_root = os.path.join(args.output_root, args.model_key, settings["exp_name"])
    os.makedirs(exp_root, exist_ok=True)

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        run_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        seed_dir = os.path.join(exp_root, f"seed={seed}_{run_time}")
        os.makedirs(seed_dir, exist_ok=False)

        args_dump = vars(args).copy()
        args_dump.update(
            {
                "resolved_num_prompts": selected_count,
                "guidance_reward_fn": args.guidance_reward_fn,
                "model_name_resolved": settings["model_name"],
                "num_inference_steps_resolved": settings["num_inference_steps"],
                "guidance_scale_resolved": settings["guidance_scale"],
                "height_resolved": settings["height"],
                "width_resolved": settings["width"],
            }
        )
        with open(os.path.join(seed_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(args_dump, f, indent=2)

        winner_rows = []
        for local_idx, item in enumerate(tqdm(selected_prompts, desc=f"seed={seed}")):
            prompt_id = args.prompt_start_id + local_idx
            prompt = item["prompt"]
            prompt_dir = os.path.join(seed_dir, f"{prompt_id:05d}")
            best_png = os.path.join(prompt_dir, "best_of_n_samples", "00000.png")
            if args.resume_skip_existing and os.path.exists(best_png):
                continue

            scorer.set_prompts([prompt])
            if args.model_key == "sd15":
                generator = torch.Generator(device=device).manual_seed(seed + prompt_id)
                latents = torch.randn((1, 4, 64, 64), generator=generator, device=device)
                images, _ = pipe(
                    [prompt],
                    num_inference_steps=settings["num_inference_steps"],
                    guidance_scale=settings["guidance_scale"],
                    height=settings["height"],
                    width=settings["width"],
                    eta=args.eta,
                    latents=latents,
                    num_images_per_prompt=1,
                    output_type="pil",
                )
                image = images[0]
                score = scorer.score_pil([prompt], [image])[0]
            elif args.model_key == "sdxl":
                image, score = run_sdxl_true_svdd(
                    pipe=pipe,
                    scorer=scorer,
                    prompt=prompt,
                    settings=settings,
                    args=args,
                    seed=seed,
                    prompt_id=prompt_id,
                    device=device,
                )
            else:
                image, score = run_sd35_true_svdd(
                    pipe=pipe,
                    scorer=scorer,
                    prompt=prompt,
                    settings=settings,
                    args=args,
                    seed=seed,
                    prompt_id=prompt_id,
                    device=device,
                )
            save_prompt_outputs(prompt_dir, item, image, score)
            winner_rows.append(
                {
                    "seed": seed,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "image_reward": float(score),
                }
            )

        write_seed_aggregates(seed_dir, winner_rows)


if __name__ == "__main__":
    main()
