# Best-of-N baseline launcher with launch_eval_runs.py-compatible output format.
import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, UNet2DConditionModel
from tqdm import tqdm

import sys

sys.path.append("fkd_diffusers")

from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion
from fkd_diffusers.fkd_pipeline_sd3 import FKDStableDiffusion3
from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL
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
    if "prompt" not in data[0]:
        assert "text" in data[0], "Prompt data should have 'prompt' or 'text' key"
        for item in data:
            item["prompt"] = item["text"]
    if max_prompts is not None:
        data = data[:max_prompts]
    return data


def make_pipeline(args):
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
        unet = UNet2DConditionModel.from_pretrained(
            "mhdang/dpo-sdxl-text2image-v1", subfolder="unet", torch_dtype=torch.float16
        )
        pipe.unet = unet
    elif "mhdang/dpo" in args.model_name and "xl" not in args.model_name:
        pipe = FKDStableDiffusion.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        unet = UNet2DConditionModel.from_pretrained(
            "mhdang/dpo-sd1.5-text2image-v1", subfolder="unet", torch_dtype=torch.float16
        )
        pipe.unet = unet
    else:
        print("Using SD")
        pipe = FKDStableDiffusion.from_pretrained(args.model_name, torch_dtype=torch.float16)

    if not isinstance(pipe, FKDStableDiffusion3):
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    prompt_data = load_geneval_metadata(args.prompt_path)
    pipe = make_pipeline(args)

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
        prompt = [item["prompt"]] * args.best_of_n
        start_time = datetime.now()

        prompt_path = os.path.join(output_dir, f"{prompt_idx:0>5}")
        os.makedirs(prompt_path, exist_ok=True)
        with open(os.path.join(prompt_path, "metadata.jsonl"), "w", encoding="utf-8") as f:
            json.dump(item, f)

        base_seed = args.seed + prompt_idx * args.best_of_n
        generators = [
            torch.Generator(device=device).manual_seed(base_seed + idx)
            for idx in range(args.best_of_n)
        ]

        if isinstance(pipe, FKDStableDiffusion3):
            output = pipe(
                prompt,
                num_inference_steps=args.num_inference_steps,
                generator=generators,
                output_type="pil",
            )
        else:
            output = pipe(
                prompt,
                num_inference_steps=args.num_inference_steps,
                eta=args.eta,
                generator=generators,
                output_type="pil",
            )
        images = output.images if hasattr(output, "images") else output[0]

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

        if args.save_individual_images:
            sample_path = os.path.join(prompt_path, "samples")
            os.makedirs(sample_path, exist_ok=True)
            for image_idx, image in enumerate(images):
                image.save(os.path.join(sample_path, f"{image_idx:05}.png"))

            best_of_n_sample_path = os.path.join(prompt_path, "best_of_n_samples")
            os.makedirs(best_of_n_sample_path, exist_ok=True)
            images[0].save(os.path.join(best_of_n_sample_path, "00000.png"))

        with open(os.path.join(prompt_path, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f)

        _, ax = plt.subplots(1, args.best_of_n, figsize=(args.best_of_n * 5, 5))
        if args.best_of_n == 1:
            ax = [ax]
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
    parser.add_argument("--best_of_n", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=64)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--guidance_reward_fn", type=str, default="ImageReward")
    parser.add_argument(
        "--metrics_to_compute",
        type=str,
        default="ImageReward#HumanPreference",
        help="# separated list of metrics",
    )
    parser.add_argument("--prompt_path", type=str, default="geneval_metadata.jsonl")
    parser.add_argument("--model_idx", type=int, default=6, help="Used for selecting model and configuration")
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.prompt_path == "geneval_metadata.jsonl":
        args.save_individual_images = True

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
    elif args.model_idx == 100:
        args.model_name = "mhdang/dpo-sd1.5-text2image-v1"
    elif args.model_idx == 101:
        args.model_name = "mhdang/dpo-sdxl-text2image-v1"
    else:
        raise ValueError(f"Unknown model index {args.model_idx}")

    if not args.output_name:
        args.output_dir = args.prompt_path.replace(".json", "_outputs")
    return args


if __name__ == "__main__":
    args = get_args()
    for seed in [42, 43, 44]:
        args.seed = seed
        main(args)
