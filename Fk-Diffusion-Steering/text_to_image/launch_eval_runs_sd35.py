# SD3.5-specific single-run evaluation script.
import argparse
import csv
import inspect
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import sys

sys.path.append("fkd_diffusers")

from fkd_diffusers.fkd_class import FKD
from fkd_diffusers.fkd_pipeline_sd3 import FKDStableDiffusion3
from fkd_diffusers.rewards import get_reward_function
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


def decode_latents_to_tensor(*, pipeline, latents: torch.Tensor) -> torch.Tensor:
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


def compute_x0_preds_sd35(*, pipeline, prev_latents: torch.Tensor, t: torch.Tensor, callback_kwargs):
    prompt_embeds = callback_kwargs["prompt_embeds"]
    pooled_prompt_embeds = callback_kwargs["pooled_prompt_embeds"]
    latent_model_input = (
        torch.cat([prev_latents] * 2) if pipeline.do_classifier_free_guidance else prev_latents
    )
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
    return prev_latents - sigma * noise_pred


def format_gamma_tag(gamma_target):
    if gamma_target is None:
        return "none"
    return f"{float(gamma_target):.3f}".replace(".", "p")


def install_sd35_step_wrapper_stochastic(scheduler, gamma_target):
    original_step = scheduler.step
    gamma_cap = np.sqrt(2.0) - 1.0
    gamma_eff = float(np.clip(float(gamma_target), 0.0, gamma_cap))
    if gamma_eff != float(gamma_target):
        print(
            "gamma_target was clamped to scheduler limit: "
            f"requested={float(gamma_target):.6f}, effective={gamma_eff:.6f}"
        )

    def wrapped_step(*step_args, **step_kwargs):
        out = original_step(*step_args, **step_kwargs)

        if gamma_eff <= 0.0:
            return out

        timesteps = getattr(scheduler, "timesteps", None)
        if timesteps is None:
            return out
        n_steps = len(timesteps)
        if n_steps <= 1:
            return out

        # Keep these fixed per requested sweep configuration.
        s_noise = 1.0
        s_tmin = 0.0
        s_tmax = float("inf")
        _ = gamma_eff * max(n_steps - 1, 0)  # explicit mapping retained for traceability

        timestep = None
        if len(step_args) >= 2:
            timestep = step_args[1]
        elif "timestep" in step_kwargs:
            timestep = step_kwargs["timestep"]
        if timestep is None:
            return out

        if hasattr(scheduler, "index_for_timestep"):
            sigma_idx = scheduler.index_for_timestep(timestep)
        else:
            ts = scheduler.timesteps
            sigma_idx = int((ts == timestep).nonzero(as_tuple=True)[0][0].item())

        sigma = scheduler.sigmas[sigma_idx]
        sigma_f = float(sigma.item()) if isinstance(sigma, torch.Tensor) else float(sigma)
        if not (s_tmin <= sigma_f <= s_tmax):
            return out

        if isinstance(out, tuple):
            prev_sample = out[0]
        else:
            prev_sample = out.prev_sample

        noise_std = sigma_f * np.sqrt(max((1.0 + gamma_eff) ** 2 - 1.0, 0.0)) * s_noise
        if noise_std <= 0.0:
            return out

        generator = step_kwargs.get("generator", None)
        if isinstance(generator, torch.Generator):
            noise = torch.randn(
                prev_sample.shape,
                generator=generator,
                device=prev_sample.device,
                dtype=prev_sample.dtype,
            )
        else:
            noise = torch.randn_like(prev_sample)

        perturbed_prev_sample = prev_sample + noise * noise_std
        if isinstance(out, tuple):
            return (perturbed_prev_sample,) + out[1:]
        out.prev_sample = perturbed_prev_sample
        return out

    scheduler.step = wrapped_step
    print(
        "Installed SD3.5 custom gamma stochastic wrapper with deterministic scheduler.step base, "
        f"gamma_target={float(gamma_target):.6f}, gamma_effective={gamma_eff:.6f}"
    )


def run_sd35_generation(pipe, prompt_list, generators, args):
    if not args.use_smc:
        output = pipe(
            prompt_list,
            num_inference_steps=args.num_inference_steps,
            generator=list(generators),
            output_type="pil",
        )
        return output.images if hasattr(output, "images") else output[0]

    reward_prompt = prompt_list
    fkd_args = dict(
        lmbda=args.lmbda,
        num_particles=args.num_particles,
        use_smc=args.use_smc,
        adaptive_resampling=args.adaptive_resampling,
        resample_frequency=args.resample_frequency,
        time_steps=args.num_inference_steps,
        resampling_t_start=args.resample_t_start,
        resampling_t_end=args.resample_t_end,
        guidance_reward_fn=args.guidance_reward_fn,
        potential_type=args.potential_type,
        resampling=args.resampling,
        tempering_schedule=args.tempering_schedule,
        debug_resampling=args.debug_resampling,
        custom_stochastic_sampling=args.custom_stochastic_sampling,
        eta=args.eta,
    )

    def reward_fn(decoded_tensor: torch.Tensor) -> torch.Tensor:
        pil_images = pipe.image_processor.postprocess(decoded_tensor, output_type="pil")
        rewards = get_reward_function(
            fkd_args["guidance_reward_fn"],
            images=pil_images,
            prompts=reward_prompt,
            metric_to_chase=fkd_args.get("metric_to_chase", None),
        )
        return torch.tensor(rewards, device=decoded_tensor.device)

    fkd = FKD(
        latent_to_decode_fn=lambda x: decode_latents_to_tensor(pipeline=pipe, latents=x),
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
            raise RuntimeError("prev_latents is not initialized")

        x0_preds = compute_x0_preds_sd35(
            pipeline=pipe,
            prev_latents=prev_latents,
            t=timestep_value,
            callback_kwargs=callback_kwargs,
        )
        latents, _ = fkd.resample(sampling_idx=step_idx, latents=latents, x0_preds=x0_preds)

        if args.custom_stochastic_sampling and args.eta > 0:
            resample_info = getattr(fkd, "last_resample_info", {})
            did_resample = bool(resample_info.get("did_resample", False))
            if did_resample:
                # Mild, annealed post-resample stochasticity: stronger early, weaker late.
                total_steps = max(args.num_inference_steps - 1, 1)
                progress = min(max(float(step_idx) / float(total_steps), 0.0), 1.0)
                anneal = np.sqrt(1.0 - progress)
                latent_std = torch.std(latents.detach().float(), unbiased=False).item()
                base_strength = 0.03
                noise_scale = args.eta * base_strength * anneal * latent_std
                if noise_scale > 0:
                    noise = torch.stack(
                        [
                            torch.randn(
                                latents[idx].shape,
                                generator=generators[idx],
                                device=latents.device,
                                dtype=latents.dtype,
                            )
                            for idx in range(args.num_particles)
                        ],
                        dim=0,
                    )
                    latents = latents + noise_scale * noise
                    if args.debug_resampling:
                        print(
                            "[FKD debug] custom_stochastic step="
                            f"{int(step_idx)} noise_scale={float(noise_scale):.6f} "
                            f"anneal={float(anneal):.4f} latent_std={float(latent_std):.6f}"
                        )

        prev_latents = latents
        return {"latents": latents}

    with torch.no_grad():
        init_latents = pipe.prepare_latents(
            batch_size=args.num_particles,
            num_channels_latents=pipe.transformer.config.in_channels,
            height=pipe.transformer.config.sample_size * pipe.vae_scale_factor,
            width=pipe.transformer.config.sample_size * pipe.vae_scale_factor,
            dtype=pipe.transformer.dtype,
            device=pipe.device,
            generator=list(generators),
            latents=None,
        )
        prev_latents = init_latents
        output = pipe(
            prompt_list,
            num_inference_steps=args.num_inference_steps,
            generator=list(generators),
            latents=init_latents,
            output_type="pil",
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_inputs,
        )

    return output.images if hasattr(output, "images") else output[0]


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.resample_t_end is None:
        args.resample_t_end = args.num_inference_steps

    if args.use_smc:
        assert args.resample_frequency > 0
        assert args.num_particles > 1

    all_prompt_data = load_geneval_metadata(args.prompt_path)
    if args.prompt_start_id < 0 or args.prompt_start_id >= len(all_prompt_data):
        raise ValueError("--prompt-start-id out of range")
    remaining = len(all_prompt_data) - args.prompt_start_id
    requested = remaining if args.num_prompts is None else args.num_prompts
    selected_count = min(remaining, requested)
    prompt_data = all_prompt_data[args.prompt_start_id : args.prompt_start_id + selected_count]
    if not prompt_data:
        raise ValueError("No prompts selected")

    print("Using SD3.5")
    try:
        pipe = FKDStableDiffusion3.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load SD3.5. Ensure your HF token is configured and you accepted model terms at "
            "https://huggingface.co/stabilityai/stable-diffusion-3.5-large"
        ) from exc

    scheduler_cls = pipe.scheduler.__class__
    scheduler_init_params = set(inspect.signature(scheduler_cls.__init__).parameters.keys())
    supports_stochastic_sampling = "stochastic_sampling" in scheduler_init_params

    scheduler_step_source = ""
    try:
        scheduler_step_source = inspect.getsource(scheduler_cls.step)
    except (OSError, TypeError):
        scheduler_step_source = ""

    has_stochastic_step_logic = (
        "stochastic_sampling" in scheduler_step_source
        and ("torch.randn_like" in scheduler_step_source or "randn_tensor" in scheduler_step_source)
    )
    scheduler_source_file = inspect.getsourcefile(scheduler_cls) or "<unknown>"

    if args.stochastic_sampling and not supports_stochastic_sampling:
        raise RuntimeError(
            "stochastic_sampling was requested, but the loaded scheduler class does not support it: "
            f"{scheduler_cls.__module__}.{scheduler_cls.__name__} from {scheduler_source_file}. "
            "This environment likely uses a diffusers build without FlowMatch stochastic sampling."
        )
    if args.stochastic_sampling and not has_stochastic_step_logic:
        raise RuntimeError(
            "stochastic_sampling was requested, but the loaded scheduler step implementation does not include "
            f"stochastic noise logic in {scheduler_source_file}."
        )

    scheduler_stochastic_sampling = (
        args.stochastic_sampling
        and not args.custom_stochastic_sampling
        and not args.use_step_wrapper_stochastic
    )
    if supports_stochastic_sampling:
        pipe.scheduler = scheduler_cls.from_config(
            pipe.scheduler.config, stochastic_sampling=scheduler_stochastic_sampling
        )
    else:
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)

    if args.stochastic_sampling:
        print(
            "Stochastic sampling is ENABLED and supported by scheduler: "
            f"{scheduler_cls.__module__}.{scheduler_cls.__name__} ({scheduler_source_file})"
        )
    if args.custom_stochastic_sampling:
        print(
            "Custom FK stochastic sampling is ENABLED; scheduler stochastic sampling is DISABLED."
        )
    if args.use_step_wrapper_stochastic:
        print(
            "Step-wrapper stochastic mode is ENABLED; scheduler stochastic sampling is forced DISABLED."
        )
        install_sd35_step_wrapper_stochastic(pipe.scheduler, args.gamma_target)

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

    with open(os.path.join(output_dir, "args.json"), "w", encoding="utf-8") as f:
        args_to_dump = vars(args).copy()
        args_to_dump["resolved_num_prompts"] = selected_count
        json.dump(args_to_dump, f, indent=4)

    metrics_to_compute = args.metrics_to_compute.split("#")
    do_eval(
        prompt=["test"],
        images=[Image.new("RGB", (224, 224))],
        metrics_to_compute=metrics_to_compute,
    )

    metrics_arr = {metric: dict(mean=0, max=0, min=0, std=0) for metric in metrics_to_compute}
    n_samples = 0
    average_time = 0.0
    per_prompt_rows = []

    for local_idx, item in enumerate(tqdm(prompt_data)):
        prompt_idx = args.prompt_start_id + local_idx
        prompt = [item["prompt"]] * args.num_particles
        start_time = datetime.now()

        prompt_path = os.path.join(output_dir, f"{prompt_idx:0>5}")
        os.makedirs(prompt_path, exist_ok=True)
        with open(os.path.join(prompt_path, "metadata.jsonl"), "w", encoding="utf-8") as f:
            json.dump(item, f)

        base_seed = args.seed + prompt_idx * args.num_particles
        generators = [
            torch.Generator(device=device).manual_seed(base_seed + idx)
            for idx in range(args.num_particles)
        ]

        images = run_sd35_generation(pipe, prompt, generators, args)

        if args.use_smc and args.debug_resampling:
            print(
                "[FKD debug no-resample] Starting paired no-resampling rollouts "
                "(stochastic on/off) from the same per-particle seeds."
            )
            resampling_steps = set(
                int(x)
                for x in np.append(
                    np.arange(args.resample_t_start, args.resample_t_end + 1, args.resample_frequency),
                    args.num_inference_steps - 1,
                ).tolist()
            )
            scheduler_cls = pipe.scheduler.__class__
            scheduler_init_params = set(inspect.signature(scheduler_cls.__init__).parameters.keys())
            supports_stochastic_sampling = "stochastic_sampling" in scheduler_init_params

            def debug_no_resample_rollout(*, stochastic_sampling):
                if stochastic_sampling and not supports_stochastic_sampling:
                    print(
                        "[FKD debug no-resample] skipping stochastic run: "
                        "scheduler does not support stochastic_sampling."
                    )
                    return

                original_scheduler = pipe.scheduler
                if supports_stochastic_sampling:
                    pipe.scheduler = scheduler_cls.from_config(
                        original_scheduler.config, stochastic_sampling=stochastic_sampling
                    )
                else:
                    pipe.scheduler = scheduler_cls.from_config(original_scheduler.config)

                debug_generators = [
                    torch.Generator(device=device).manual_seed(base_seed + idx)
                    for idx in range(args.num_particles)
                ]
                with torch.no_grad():
                    init_latents = pipe.prepare_latents(
                        batch_size=args.num_particles,
                        num_channels_latents=pipe.transformer.config.in_channels,
                        height=pipe.transformer.config.sample_size * pipe.vae_scale_factor,
                        width=pipe.transformer.config.sample_size * pipe.vae_scale_factor,
                        dtype=pipe.transformer.dtype,
                        device=pipe.device,
                        generator=list(debug_generators),
                        latents=None,
                    )

                prev_latents = init_latents
                run_label = "stochastic_on" if stochastic_sampling else "stochastic_off"
                callback_inputs = [
                    "latents",
                    "prompt_embeds",
                    "negative_prompt_embeds",
                    "pooled_prompt_embeds",
                    "negative_pooled_prompt_embeds",
                ]

                def callback_on_step_end(_pipeline, step_idx, timestep_value, callback_kwargs):
                    nonlocal prev_latents
                    latents = callback_kwargs["latents"]
                    x0_preds = compute_x0_preds_sd35(
                        pipeline=pipe,
                        prev_latents=prev_latents,
                        t=timestep_value,
                        callback_kwargs=callback_kwargs,
                    )
                    if int(step_idx) in resampling_steps:
                        decoded_tensor = decode_latents_to_tensor(pipeline=pipe, latents=x0_preds)
                        pil_images = pipe.image_processor.postprocess(decoded_tensor, output_type="pil")
                        rewards = get_reward_function(
                            args.guidance_reward_fn,
                            images=pil_images,
                            prompts=prompt,
                            metric_to_chase=None,
                        )
                        print(
                            "[FKD debug no-resample] "
                            f"mode={run_label} step={int(step_idx)} rewards={list(rewards)}"
                        )
                    prev_latents = latents
                    return {"latents": latents}

                try:
                    with torch.no_grad():
                        pipe(
                            prompt,
                            num_inference_steps=args.num_inference_steps,
                            generator=list(debug_generators),
                            latents=init_latents.clone(),
                            output_type="pil",
                            callback_on_step_end=callback_on_step_end,
                            callback_on_step_end_tensor_inputs=callback_inputs,
                        )
                finally:
                    pipe.scheduler = original_scheduler

            debug_no_resample_rollout(stochastic_sampling=True)
            debug_no_resample_rollout(stochastic_sampling=False)

        end_time = datetime.now()

        results = do_eval(prompt=prompt, images=images, metrics_to_compute=metrics_to_compute)
        time_taken = end_time - start_time
        results["time_taken"] = time_taken.total_seconds()
        results["prompt"] = prompt
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

        per_prompt_rows.append(
            {
                "gamma_target": (
                    float(args.gamma_target) if args.gamma_target is not None else ""
                ),
                "seed": int(args.seed),
                "prompt_index": int(prompt_idx),
                "prompt": item["prompt"],
                "image_reward": float(results["ImageReward"]["mean"]),
                "human_preference": float(results["HumanPreference"]["mean"]),
                "time_taken_s": float(results["time_taken"]),
            }
        )

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

        _, ax = plt.subplots(1, args.num_particles, figsize=(args.num_particles * 5, 5))
        for i, image in enumerate(images):
            ax[i].imshow(image)
            ax[i].axis("off")
        plt.suptitle(prompt[0])
        plt.savefig(os.path.join(prompt_path, "grid.png"))
        plt.close()

    for metric in metrics_to_compute:
        metrics_arr[metric]["mean"] /= n_samples
        metrics_arr[metric]["max"] /= n_samples
        metrics_arr[metric]["min"] /= n_samples
        metrics_arr[metric]["std"] /= n_samples

    with open(os.path.join(output_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_arr, f)

    csv_path = os.path.join(
        output_dir, f"per_prompt_metrics_gamma_{format_gamma_tag(args.gamma_target)}.csv"
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "gamma_target",
                "seed",
                "prompt_index",
                "prompt",
                "image_reward",
                "human_preference",
                "time_taken_s",
            ],
        )
        writer.writeheader()
        writer.writerows(per_prompt_rows)
    print(f"Saved per-prompt metrics to {csv_path}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="geneval_outputs")
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Exact output path for this run (created with exist_ok=False).",
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
    parser.add_argument("--model_idx", type=int, default=18, help="Used for selecting particle count")
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--lmbda", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adaptive_resampling", action="store_true")
    parser.add_argument("--resample_frequency", type=int, default=5)
    parser.add_argument("--resample_t_start", type=int, default=5)
    parser.add_argument("--resample_t_end", type=int, default=30)
    parser.add_argument("--potential_type", type=str, default="diff")
    parser.add_argument("--resampling", type=str, default="multinomial")
    parser.add_argument("--tempering_schedule", type=str, default="constant")
    parser.add_argument("--prompt-start-id", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--stochastic_sampling", action="store_true")
    parser.add_argument("--custom_stochastic_sampling", action="store_true")
    parser.add_argument("--use_step_wrapper_stochastic", action="store_true")
    parser.add_argument("--gamma_target", type=float, default=None)
    parser.add_argument("--debug_resampling", action="store_true")
    parser.add_argument(
        "--single_seed_mode",
        action="store_true",
        help="Run only the provided --seed instead of the default [42, 43, 44] sweep.",
    )

    args = parser.parse_args()
    print(args.adaptive_resampling)
    if not (0.0 <= args.eta <= 1.0):
        raise ValueError(f"eta must be in [0, 1], got {args.eta}.")
    if args.stochastic_sampling and args.custom_stochastic_sampling:
        raise ValueError(
            "stochastic_sampling and custom_stochastic_sampling are mutually exclusive."
        )
    if args.use_step_wrapper_stochastic and not args.stochastic_sampling:
        raise ValueError(
            "use_step_wrapper_stochastic requires stochastic_sampling=True."
        )
    if args.use_step_wrapper_stochastic and args.gamma_target is None:
        raise ValueError(
            "gamma_target must be provided when use_step_wrapper_stochastic is enabled."
        )

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

    if args.model_idx not in [16, 17, 18, 19]:
        raise ValueError(
            f"launch_eval_runs_sd35.py is SD3.5-specific. Use model_idx in [16,17,18,19], got {args.model_idx}."
        )
    args.model_name = "stabilityai/stable-diffusion-3.5-large"
    if not args.output_name:
        args.output_dir = args.prompt_path.replace(".json", "_outputs")
    return args


if __name__ == "__main__":
    args = get_args()
    if args.single_seed_mode:
        main(args)
    else:
        for seed in [42, 43, 44]:
            args.seed = seed
            main(args)
