#!/usr/bin/env python3
"""Visualize a diffusion denoising trajectory for a single prompt.

Top row:    noisy latent x_t decoded to an image.
Bottom row: model's clean-image estimate x_hat_0 decoded to an image.
Columns:    selected denoising steps.

Column titles show "step {s}\n({pct}% denoised)" (step 0 is "pure noise");
under each bottom image we print the ImageReward of the clean-image estimate.

Everything mirrors the reference reward-collection code
(``Fk-Diffusion-Steering/text_to_image/collect_image_reward_signal.py``):
  * SD1.5 / SDXL use the DDIM x0 = ``pred_original_sample`` path (64 steps).
  * SD3.5 uses the flow-matching estimate x0 = x_t - sigma * v with its native
    FlowMatchEuler scheduler (32 steps).
For SD3.5 on a 24 GB card we lean on the pipeline's native VRAM helpers
(model CPU offload -> keeps the T5/CLIP text encoders on CPU, VAE slicing/tiling).
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from diffusers import DDIMScheduler

REPO = Path(__file__).resolve().parents[2]
TEXT_TO_IMAGE = REPO / "Fk-Diffusion-Steering" / "text_to_image"
sys.path.insert(0, str(TEXT_TO_IMAGE))
# fkd_class imports `smc_utils` as a top-level module; it lives under fkd_diffusers.
sys.path.insert(0, str(TEXT_TO_IMAGE / "fkd_diffusers"))

from collect_image_reward_signal import (  # noqa: E402
    build_pipeline,
    compute_x0_preds_sd35,
    compute_x0_preds_non_sd35,
    decode_latents_to_tensor_sd35,
)
from fkd_diffusers.fkd_pipeline_sd import latent_to_decode as latent_to_decode_sd  # noqa: E402
from fkd_diffusers.fkd_pipeline_sdxl import (  # noqa: E402
    FKDStableDiffusionXL,
    latent_to_decode as latent_to_decode_sdxl,
)
from fkd_diffusers.rewards import do_image_reward  # noqa: E402


# key -> (model-name passed to build_pipeline, default #inference steps)
MODEL_PRESETS = {
    "sd15": ("stable-diffusion-v1-5", 64),
    "sdxl": ("stable-diffusion-xl", 64),
    "sd35": ("stable-diffusion-3.5-large", 32),
}
MODEL_LABELS = {"sd15": "SD 1.5", "sdxl": "SDXL", "sd35": "SD 3.5"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Denoising trajectory visualization.")
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="sdxl")
    parser.add_argument("--prompt", type=str, default="a photo of a cow left of a stop sign")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Overrides the per-model default (64 for SD1.5/SDXL, 32 for SD3.5).",
    )
    parser.add_argument(
        "--capture-steps",
        type=str,
        default="",
        help="Comma-separated steps to visualize. Empty = [0, T/4, T/2, T].",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Offload models to CPU between forwards (default: on for SD3.5 only).",
    )
    parser.add_argument("--output-dir", type=str, default="paper_figures/figures")
    parser.add_argument("--base-name", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def score_ir(prompt, pil_image) -> float:
    return float(do_image_reward(images=[pil_image], prompts=[prompt])[0])


def main() -> None:
    args = parse_args()
    # Inference only: no autograd graphs anywhere (the standalone VAE decodes
    # below run outside the pipeline's own no_grad block and would otherwise OOM).
    torch.set_grad_enabled(False)

    model_name, default_steps = MODEL_PRESETS[args.model]
    total_steps = int(args.num_inference_steps or default_steps)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpu_offload = args.cpu_offload if args.cpu_offload is not None else (args.model == "sd35")

    if args.capture_steps.strip():
        capture_steps = sorted({int(s) for s in args.capture_steps.split(",") if s.strip()})
    else:
        capture_steps = sorted(
            {0, round(total_steps * 0.25), round(total_steps * 0.5), total_steps}
        )

    # When offloading, build on CPU first: build_pipeline() otherwise does a
    # blanket .to("cuda") that would try to fit MMDiT + all text encoders (incl.
    # T5) + VAE on the GPU at once and OOM before offload can help.
    build_device = "cpu" if cpu_offload else device
    pipeline, is_sd35 = build_pipeline(model_name, build_device)
    pipeline.set_progress_bar_config(disable=False)

    # --- VRAM optimizations (the pipelines expose these natively) ---
    if cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
        # Streams one module at a time to the GPU for its forward pass and parks
        # the rest on CPU. The offload chain (text encoders -> transformer -> vae)
        # guarantees the text encoders (the big T5) sit on CPU during sampling and
        # that the transformer is evicted before the VAE runs, so MMDiT and VAE
        # are never both fully resident.
        pipeline.enable_model_cpu_offload(device=device)
    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    if hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()

    exec_device = getattr(pipeline, "_execution_device", pipeline.device)

    # SD1.5/SDXL: DDIM x0 estimate needs a scheduler; use a dedicated copy so we
    # never mutate the live sampling scheduler. SD3.5 reads sigmas directly.
    callback_scheduler = None
    if not is_sd35:
        callback_scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        callback_scheduler.set_timesteps(total_steps, device=exec_device)

    core = pipeline.transformer if is_sd35 else pipeline.unet
    height = width = core.config.sample_size * pipeline.vae_scale_factor

    def decode_to_pil(latents):
        if is_sd35:
            # Our manual x0 decode calls vae.decode() directly, which bypasses the
            # accelerate offload hook. So do the eviction by hand: park MMDiT on
            # CPU and pull the VAE onto the GPU, so the two never co-reside. The
            # hook reloads the transformer on the next sampling step.
            if cpu_offload:
                pipeline.transformer.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pipeline.vae.to(exec_device)
            img = decode_latents_to_tensor_sd35(pipeline=pipeline, latents=latents)
        else:
            latent_fn = (
                latent_to_decode_sdxl
                if isinstance(pipeline, FKDStableDiffusionXL)
                else latent_to_decode_sd
            )
            img = latent_fn(model=pipeline, output_type="pt", latents=latents)
        return pipeline.image_processor.postprocess(img, output_type="pil")[0]

    generator = torch.Generator(device=exec_device).manual_seed(int(args.seed))
    prev_latents = pipeline.prepare_latents(
        1,
        core.config.in_channels,
        height,
        width,
        core.dtype,
        exec_device,
        generator,
        None,
    )

    # step -> {"noisy": PIL, "x0": PIL, "ir": float}
    captured: dict[int, dict] = {}

    # Step 0 = pure noise: decode the initial latent; the callback for step_idx=0
    # fills the matching x0 estimate.
    if 0 in capture_steps:
        captured[0] = {"noisy": decode_to_pil(prev_latents)}

    def on_step(_pipe, step_idx, t, callback_kwargs):
        nonlocal prev_latents
        # prev_latents here is x_t going *into* step step_idx (step_idx steps done).
        if step_idx in capture_steps:
            # Do the transformer/UNet work (x0 estimate) before any VAE decode so
            # model-offload only swaps transformer<->vae once per captured step.
            if is_sd35:
                x0 = compute_x0_preds_sd35(
                    pipeline=pipeline,
                    prev_latents=prev_latents,
                    t=t,
                    callback_kwargs=callback_kwargs,
                )
            else:
                x0 = compute_x0_preds_non_sd35(
                    pipeline=pipeline,
                    scheduler=callback_scheduler,
                    prev_latents=prev_latents,
                    t=t,
                    callback_kwargs=callback_kwargs,
                    eta=args.eta,
                )
            captured.setdefault(step_idx, {})
            if step_idx != 0:
                captured[step_idx]["noisy"] = decode_to_pil(prev_latents)
            x0_pil = decode_to_pil(x0)
            captured[step_idx]["x0"] = x0_pil
            captured[step_idx]["ir"] = score_ir(args.prompt, x0_pil)
        prev_latents = callback_kwargs.get("latents", prev_latents)
        return {}

    if is_sd35:
        callback_inputs = [
            "latents",
            "prompt_embeds",
            "negative_prompt_embeds",
            "pooled_prompt_embeds",
            "negative_pooled_prompt_embeds",
        ]
        call_kwargs = {}
    elif isinstance(pipeline, FKDStableDiffusionXL):
        callback_inputs = [
            "latents",
            "prompt_embeds",
            "negative_prompt_embeds",
            "add_text_embeds",
            "negative_pooled_prompt_embeds",
            "add_time_ids",
            "negative_add_time_ids",
        ]
        call_kwargs = {"eta": args.eta}
    else:  # SD1.5
        callback_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
        call_kwargs = {"eta": args.eta}

    out = pipeline(
        args.prompt,
        num_inference_steps=total_steps,
        generator=generator,
        latents=prev_latents,
        output_type="pil",
        callback_on_step_end=on_step,
        callback_on_step_end_tensor_inputs=callback_inputs,
        **call_kwargs,
    )

    # Fully-denoised step (== total_steps): both rows show the final image.
    final_image = out.images[0]
    if total_steps in capture_steps:
        final_ir = score_ir(args.prompt, final_image)
        captured[total_steps] = {"noisy": final_image, "x0": final_image, "ir": final_ir}

    # --- Build the figure ---
    cols = [s for s in capture_steps if s in captured]
    n = len(cols)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 6.4))
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, step in enumerate(cols):
        top_ax = axes[0, j]
        bot_ax = axes[1, j]
        top_ax.imshow(captured[step]["noisy"])
        bot_ax.imshow(captured[step]["x0"])
        for ax in (top_ax, bot_ax):
            ax.set_xticks([])
            ax.set_yticks([])

        if step == 0:
            title = "step 0\npure noise"
        else:
            pct = round(step / total_steps * 100)
            title = f"step {step}\n({pct}% denoised)"
        top_ax.set_title(title, fontsize=12)

        ir = captured[step].get("ir")
        bot_ax.set_xlabel(f"IR = {ir:.3f}" if ir is not None else "IR = n/a", fontsize=12)

    axes[0, 0].set_ylabel("Noisy image  $x_t$", fontsize=12)
    axes[1, 0].set_ylabel(r"Clean estimate  $\hat{x}_0$", fontsize=12)

    fig.suptitle(
        f'"{args.prompt}"  ({MODEL_LABELS[args.model]}, {total_steps} steps)', fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    base_name = args.base_name or f"denoising_trajectory_{args.model}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{base_name}.png"
    pdf_path = output_dir / f"{base_name}.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    for step in cols:
        ir = captured[step].get("ir")
        print(f"  step {step:>3}: IR(x0_hat) = {ir:.4f}" if ir is not None else f"  step {step}: n/a")


if __name__ == "__main__":
    main()
