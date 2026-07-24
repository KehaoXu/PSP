import argparse
import datetime
import inspect
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler
from tqdm import tqdm

from aesthetic_scorer import AestheticScorerDiff, AestheticScorerDiff_Time, MLPDiff
from compressibility_scorer import CompressibilityScorerDiff, CompressibilityScorer_modified, jpeg_compressibility
from dataset import AVACompressibilityDataset, AVACLIPDataset
from imagereward_adapter import ImageRewardScorerAdapter
from run_logger import RUN_LOGGER as wandb
from sd_pipeline import Decoding_nonbatch_SDPipeline


def parse():
    parser = argparse.ArgumentParser(description="SVDD PM single-prompt GenEval inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_key", type=str, default="sd15", choices=["sd15", "sdxl", "sd35"])
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--reward", type=str, default="aesthetic")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--val_bs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duplicate_size", type=int, default=20)
    parser.add_argument("--variant", type=str, default="PM")
    parser.add_argument("--valuefunction", type=str, default="")
    parser.add_argument("--metadata_path", type=str, default="")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--prompt_override", type=str, default="")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--stochastic_sampling", action="store_true")
    parser.add_argument("--use_step_wrapper_stochastic", action="store_true")
    parser.add_argument("--gamma_target", type=float, default=None)
    return parser.parse_args()


def _default_geneval_metadata_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "Fk-Diffusion-Steering" / "text_to_image" / "prompt_files" / "geneval_metadata.jsonl"


def load_geneval_prompt(metadata_path: str, prompt_index: int) -> str:
    if prompt_index < 0:
        raise ValueError(f"prompt_index must be >= 0, got {prompt_index}")

    selected_prompt = None
    with open(metadata_path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx == prompt_index:
                payload = json.loads(line)
                if "prompt" not in payload:
                    raise KeyError(f"Missing 'prompt' key in metadata line {idx}")
                selected_prompt = payload["prompt"]
                break

    if selected_prompt is None:
        raise IndexError(f"prompt_index={prompt_index} out of range for metadata file: {metadata_path}")
    return selected_prompt


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


def install_sd35_step_wrapper_stochastic(scheduler, gamma_target):
    gamma_cap = np.sqrt(2.0) - 1.0
    gamma_eff = float(np.clip(float(gamma_target), 0.0, gamma_cap))
    original_step = scheduler.step

    def wrapped_step(*step_args, **step_kwargs):
        out = original_step(*step_args, **step_kwargs)

        if gamma_eff <= 0.0:
            return out

        timesteps = getattr(scheduler, "timesteps", None)
        if timesteps is None or len(timesteps) <= 1:
            return out

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
        noise_std = sigma_f * np.sqrt(max((1.0 + gamma_eff) ** 2 - 1.0, 0.0))
        if noise_std <= 0.0:
            return out

        if isinstance(out, tuple):
            prev_sample = out[0]
        else:
            prev_sample = out.prev_sample

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


def resolve_backbone_settings(args):
    defaults = BACKBONE_DEFAULTS[args.model_key]
    settings = {
        "model_name": args.model_name.strip() if args.model_name.strip() else defaults["model_name"],
        "num_inference_steps": args.num_inference_steps if args.num_inference_steps is not None else defaults["num_inference_steps"],
        "guidance_scale": args.guidance_scale if args.guidance_scale is not None else defaults["guidance_scale"],
        "height": args.height if args.height is not None else defaults["height"],
        "width": args.width if args.width is not None else defaults["width"],
        "torch_dtype": defaults["torch_dtype"],
    }
    return settings


def load_generation_pipeline(args, settings):
    repo_root = Path(__file__).resolve().parents[1]
    fk_text_to_image_dir = repo_root / "Fk-Diffusion-Steering" / "text_to_image"
    if str(fk_text_to_image_dir) not in sys.path:
        sys.path.append(str(fk_text_to_image_dir))

    model_key = args.model_key
    if model_key == "sd15":
        pipe = Decoding_nonbatch_SDPipeline.from_pretrained(
            settings["model_name"], local_files_only=True
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        return pipe

    if model_key == "sdxl":
        from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL

        pipe = FKDStableDiffusionXL.from_pretrained(
            settings["model_name"], torch_dtype=settings["torch_dtype"]
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        if hasattr(pipe.vae, "config") and hasattr(pipe.vae.config, "force_upcast"):
            pipe.vae.config.force_upcast = True
        return pipe

    from fkd_diffusers.fkd_pipeline_sd3 import FKDStableDiffusion3

    pipe = FKDStableDiffusion3.from_pretrained(
        settings["model_name"], torch_dtype=settings["torch_dtype"]
    )
    scheduler_cls = pipe.scheduler.__class__
    init_params = set(inspect.signature(scheduler_cls.__init__).parameters.keys())
    supports_stochastic = "stochastic_sampling" in init_params
    scheduler_stochastic = args.stochastic_sampling and not args.use_step_wrapper_stochastic

    if supports_stochastic:
        pipe.scheduler = scheduler_cls.from_config(
            pipe.scheduler.config,
            stochastic_sampling=scheduler_stochastic,
        )
    else:
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)

    if args.use_step_wrapper_stochastic:
        gamma_target = 0.005 if args.gamma_target is None else args.gamma_target
        install_sd35_step_wrapper_stochastic(pipe.scheduler, gamma_target)
    return pipe


args = parse()
args.reward = args.reward.lower()
device = args.device
save_file = True
settings = resolve_backbone_settings(args)
if args.model_key == "sd35" and args.gamma_target is None:
    args.gamma_target = 0.005

if args.seed > 0:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.model_key == "sd15":
        shape = (args.num_images // args.bs, args.bs, 4, 64, 64)
        init_latents = torch.randn(shape, device=device)
    else:
        init_latents = None
else:
    init_latents = None

run_name = f"{args.variant}_M={args.duplicate_size}_{args.valuefunction.split('/')[-1] if args.valuefunction != '' else ''}"
unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
run_name = run_name + "_" + unique_id + f"_geneval_single_{args.model_key}"

if args.out_dir == "":
    args.out_dir = "logs/" + run_name
os.makedirs(args.out_dir, exist_ok=True)

if args.num_images % args.bs != 0:
    raise ValueError(f"num_images ({args.num_images}) must be divisible by bs ({args.bs})")

metadata_path = args.metadata_path.strip() if args.metadata_path else str(_default_geneval_metadata_path())
if args.prompt_override.strip():
    selected_prompt = args.prompt_override.strip()
else:
    selected_prompt = load_geneval_prompt(metadata_path, args.prompt_index)

print(f"Using prompt: {selected_prompt}")
print(f"Metadata path: {metadata_path}")
print(f"Prompt index: {args.prompt_index}")
print(
    "Backbone settings: "
    f"model_key={args.model_key}, model_name={settings['model_name']}, "
    f"steps={settings['num_inference_steps']}, cfg={settings['guidance_scale']}, "
    f"size={settings['height']}x{settings['width']}"
)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
initial_memory = torch.cuda.memory_allocated()

sd_model = load_generation_pipeline(args, settings)
sd_model.to(device)

if hasattr(sd_model, "vae") and sd_model.vae is not None:
    sd_model.vae.requires_grad_(False)
if hasattr(sd_model, "text_encoder") and sd_model.text_encoder is not None:
    sd_model.text_encoder.requires_grad_(False)
if hasattr(sd_model, "text_encoder_2") and sd_model.text_encoder_2 is not None:
    sd_model.text_encoder_2.requires_grad_(False)
if hasattr(sd_model, "text_encoder_3") and sd_model.text_encoder_3 is not None:
    sd_model.text_encoder_3.requires_grad_(False)
if hasattr(sd_model, "unet") and sd_model.unet is not None:
    sd_model.unet.requires_grad_(False)
if hasattr(sd_model, "transformer") and sd_model.transformer is not None:
    sd_model.transformer.requires_grad_(False)

if hasattr(sd_model, "vae") and sd_model.vae is not None:
    sd_model.vae.eval()
if hasattr(sd_model, "text_encoder") and sd_model.text_encoder is not None:
    sd_model.text_encoder.eval()
if hasattr(sd_model, "text_encoder_2") and sd_model.text_encoder_2 is not None:
    sd_model.text_encoder_2.eval()
if hasattr(sd_model, "text_encoder_3") and sd_model.text_encoder_3 is not None:
    sd_model.text_encoder_3.eval()
if hasattr(sd_model, "unet") and sd_model.unet is not None:
    sd_model.unet.eval()
if hasattr(sd_model, "transformer") and sd_model.transformer is not None:
    sd_model.transformer.eval()

assert args.variant in ["PM", "MC"]

if args.reward == "compressibility":
    if args.variant == "PM":
        scorer = CompressibilityScorer_modified(dtype=torch.float32)
    elif args.variant == "MC":
        scorer = CompressibilityScorerDiff(dtype=torch.float32).to(device)
elif args.reward == "aesthetic":
    if args.variant == "PM":
        scorer = AestheticScorerDiff(dtype=torch.float32).to(device)
    elif args.variant == "MC":
        scorer = AestheticScorerDiff_Time(dtype=torch.float32).to(device)
        if args.valuefunction != "":
            scorer.set_valuefunction(args.valuefunction)
            scorer = scorer.to(device)
elif args.reward.lower() in {"imagereward", "image_reward", "imagereward-v1.0"}:
    scorer = ImageRewardScorerAdapter(device=device)
else:
    raise ValueError("Invalid reward")

if hasattr(scorer, "requires_grad_"):
    scorer.requires_grad_(False)
if hasattr(scorer, "eval"):
    scorer.eval()

if args.model_key == "sd15":
    sd_model.setup_scorer(scorer)
    sd_model.set_variant(args.variant)
    sd_model.set_reward(args.reward)
    sd_model.set_parameters(args.bs, args.duplicate_size)
else:
    print(
        f"Running backbone '{args.model_key}' with diffusion defaults; "
        "SVDD PM duplicate scoring controls are not applied for this backbone."
    )

image = []
eval_prompt_list = []

for i in tqdm(range(args.num_images // args.bs), desc="Generating Images"):
    wandb.log({"inner_iter": i})
    init_i = None if init_latents is None else init_latents[i]
    eval_prompts = [selected_prompt for _ in range(args.bs)]
    eval_prompt_list.extend(eval_prompts)
    if hasattr(scorer, "set_prompts"):
        scorer.set_prompts(eval_prompts)

    common_kwargs = dict(
        prompt=eval_prompts,
        num_inference_steps=settings["num_inference_steps"],
        guidance_scale=settings["guidance_scale"],
        height=settings["height"],
        width=settings["width"],
        num_images_per_prompt=1,
        output_type="pil",
    )
    if args.model_key == "sd15":
        common_kwargs["eta"] = 1.0
        common_kwargs["latents"] = init_i
        image_, _ = sd_model(**common_kwargs)
    else:
        output = sd_model(**common_kwargs)
        image_ = output.images if hasattr(output, "images") else output[0]
    image.extend(image_)

end_event.record()
torch.cuda.synchronize()
gpu_time = start_event.elapsed_time(end_event) / 1000
max_memory = torch.cuda.max_memory_allocated()
max_memory_used = (max_memory - initial_memory) / (1024 ** 2)

wandb.log(
    {
        "GPUTimeInS": gpu_time,
        "MaxMemoryInMb": max_memory_used,
    }
)

if args.reward == "compressibility":
    gt_dataset = AVACompressibilityDataset(image)
elif args.reward == "aesthetic":
    from importlib import resources

    assets_path = resources.files("assets")
    eval_model = MLPDiff().to(device)
    eval_model.requires_grad_(False)
    eval_model.eval()
    state_dict = torch.load(
        assets_path.joinpath("sac+logos+ava1-l14-linearMSE.pth"),
        map_location=device,
        weights_only=True,
    )
    eval_model.load_state_dict(state_dict)
    gt_dataset = AVACLIPDataset(image)
elif args.reward.lower() in {"imagereward", "image_reward", "imagereward-v1.0"}:
    scores = scorer.score_pil(eval_prompt_list, image)
    eval_rewards = torch.tensor(scores, dtype=torch.float32)
    gt_dataset = None

gt_dataloader = (
    torch.utils.data.DataLoader(gt_dataset, batch_size=args.val_bs, shuffle=False)
    if gt_dataset is not None
    else None
)

with torch.no_grad():
    if gt_dataloader is not None:
        eval_rewards = []

        for inputs in gt_dataloader:
            inputs = inputs.to(device)

            if args.reward == "compressibility":
                jpeg_compressibility_scores = jpeg_compressibility(inputs)
                scores = torch.tensor(jpeg_compressibility_scores, dtype=inputs.dtype, device=inputs.device)
            elif args.reward == "aesthetic":
                scores = eval_model(inputs)
                scores = scores.squeeze(1)

            eval_rewards.extend(scores.tolist())

        eval_rewards = torch.tensor(eval_rewards)

    print(f"eval_{args.reward}_rewards_mean", torch.mean(eval_rewards))
    wandb.log(
        {
            f"eval_{args.reward}_rewards_mean": torch.mean(eval_rewards),
        }
    )

if save_file:
    images = []
    log_dir = os.path.join(args.out_dir, "eval_vis")
    os.makedirs(log_dir, exist_ok=True)
    np.save(f"{args.out_dir}/scores.npy", eval_rewards)

    def save_array_to_text_file(array, file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            array_str = ",".join(map(str, array.tolist()))
            file.write(array_str + ",")

    save_array_to_text_file(eval_rewards, f"{args.out_dir}/eval_rewards.txt")
    print("Arrays have been saved to text files.")

    for idx, im in enumerate(image):
        prompt = eval_prompt_list[idx]
        reward = eval_rewards[idx]

        im.save(f"{log_dir}/{idx:03d}_{prompt}_score={reward:2f}.png")

        pil = im.resize((256, 256))
        images.append(wandb.Image(pil, caption=f"{prompt:.25} | score:{reward:.2f}"))

    wandb.log({"images": images})
