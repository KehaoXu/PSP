from typing import Dict, Literal, Tuple, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect
import os

import numpy as np
import torch
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
)

from rbf.utils.extra_utils import (
    ignore_kwargs,
)
from rbf.utils.print_utils import print_info, print_warning, print_error

import rbf.shared_modules as sm
from rbf.prior.base import Prior, NEGATIVE_PROMPT


def _sd_debug_enabled() -> bool:
    return os.environ.get("FITS_SD_DEBUG", "").strip() not in ("", "0", "false", "False")


def _sd_debug_log(message: str) -> None:
    if not _sd_debug_enabled():
        return
    try:
        debug_dir = getattr(sm.logger, "debug_dir", None)
    except Exception:
        debug_dir = None
    if not debug_dir:
        return
    try:
        with open(os.path.join(debug_dir, "sd_diag.txt"), "a") as f:
            f.write(message.rstrip() + "\n")
    except Exception:
        pass


class StableDiffusionPrior(Prior):
    @ignore_kwargs
    @dataclass
    class Config:
        device: int = 0
        batch_size: int = 1
        model_name: str = "runwayml/stable-diffusion-v1-5"
        text_prompt: str = (
            "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
        )
        negative_prompt: str = ""
        width: int = 512
        height: int = 512
        guidance_scale: int = 7.5
        root_dir: str = "./results/default"
        max_steps: int = 50

        minibatch_size: int = 10
        eta: float = 1.0

        use_dpo: bool = True
        precision: str = "fp16"
        sd_model: str = ""


    def __init__(self, cfg):
        super().__init__()
        self.cfg = self.Config(**cfg)

        print("Loading ", self.cfg.model_name)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.model_name, subfolder="scheduler"
        )

        # -------------------------------------------------------------------
        # Precision
        # -------------------------------------------------------------------
        if self.cfg.precision == "fp32":
            _dtype = torch.float32

        elif self.cfg.precision == "fp16":
            _dtype = torch.float16

        else:
            raise NotImplementedError("Only fp32 and fp16 are supported")
        

        # -------------------------------------------------------------------
        # Load the UNet model
        # -------------------------------------------------------------------
        print_info("Loading Stable Diffusion", self.cfg.sd_model, self.cfg.precision)
        if self.cfg.sd_model == "sd15":
            model_id = "runwayml/stable-diffusion-v1-5"
            unet_id = "mhdang/dpo-sd1.5-text2image-v1"
            self.cfg.guidance_scale = 7.5
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                scheduler=self.scheduler,
                torch_dtype=_dtype,
            ).to(self.cfg.device)

        elif self.cfg.sd_model == "sd2":
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.cfg.model_name,
                scheduler=self.scheduler,
                torch_dtype=_dtype,
            ).to(self.cfg.device)
            
        elif self.cfg.sd_model == "sdxl":
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            unet_id = "mhdang/dpo-sdxl-text2image-v1"
            self.cfg.guidance_scale = 7.5
            # CRITICAL: pass the DDIM scheduler so pipeline.prepare_latents uses
            # DDIM's init_noise_sigma=1.0. The SDXL default scheduler is
            # EulerDiscreteScheduler whose init_noise_sigma ~= 14.6 (sqrt(sigma_max^2 + 1)),
            # which would scale the initial random noise to ~14.6x the magnitude
            # the DDIM math expects. That scale mismatch puts the latent far out
            # of the UNet's training distribution at every timestep, makes eps
            # predictions nearly identical for any conditioning (CFG diff ~ 0),
            # and yields pure-noise outputs.
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                scheduler=self.scheduler,
                torch_dtype=_dtype,
            ).to(self.cfg.device)

        else:
            raise NotImplementedError("Only sd15 and sdxl are supported")
        

        # if self.cfg.use_dpo:
        #     print_info("Loading Diffusion-DPO")
        #     unet = UNet2DConditionModel.from_pretrained(
        #         unet_id, subfolder="unet", torch_dtype=self.pipeline.dtype,
        #     ).to(self.cfg.device)
        #     self.pipeline.unet = unet

        self.scheduler.set_timesteps(self.cfg.max_steps)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.cfg.device).to(self.pipeline.dtype)

        self.pipeline.unet.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)

        self.pipeline.unet.eval()
        self.pipeline.vae.eval()
        self.pipeline.text_encoder.eval()

        self.nfe = 0

        
    @property
    def rgb_res(self):
        return 1, 3, 512, 512
    
    @property
    def latent_res(self):
        return 1, 4, 64, 64

    def prepare_cond(self, text_prompt=None, negative_prompt=None, _pass=False):
        if not _pass:
            if hasattr(self, "cond"):
                return self.cond 
        
        text_prompt = text_prompt if text_prompt is not None else self.cfg.text_prompt
        negative_prompt = negative_prompt if negative_prompt is not None else self.cfg.negative_prompt

        print_info("Encoding text prompt", text_prompt)

        if self.cfg.sd_model == "sdxl":
            with torch.inference_mode():
                pe, ne, ppe, npe = self.pipeline.encode_prompt(
                    prompt=[text_prompt],
                    prompt_2=None,
                    device=self.pipeline.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=[negative_prompt],
                    negative_prompt_2=None,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    pooled_prompt_embeds=None,
                    negative_pooled_prompt_embeds=None,
                    lora_scale=None,
                    clip_skip=None,
                )
            # Keep neg / pos separate so the predict() CFG batch can be assembled
            # as [neg-block, pos-block] to match `cat([latents] * 2)` ordering.
            neg_prompt_embeds = ne.to(self.pipeline.device)
            pos_prompt_embeds = pe.to(self.pipeline.device)
            neg_pooled_embeds = npe.to(self.pipeline.device)
            pos_pooled_embeds = ppe.to(self.pipeline.device)
            text_encoder_projection_dim = (
                int(ppe.shape[-1])
                if self.pipeline.text_encoder_2 is None
                else self.pipeline.text_encoder_2.config.projection_dim
            )
            add_time_ids_single = self.pipeline._get_add_time_ids(
                (self.cfg.height, self.cfg.width),
                (0, 0),
                (self.cfg.height, self.cfg.width),
                dtype=pos_prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            ).to(self.pipeline.device)

            self.cond = {
                # legacy interleaved tensors (kept for backward compatibility, NOT used by predict)
                "prompt_embeds": torch.cat([neg_prompt_embeds, pos_prompt_embeds], dim=0),
                "add_text_embeds": torch.cat([neg_pooled_embeds, pos_pooled_embeds], dim=0),
                "add_time_ids": torch.cat([add_time_ids_single, add_time_ids_single], dim=0),
                # canonical neg/pos tensors used by predict()
                "neg_prompt_embeds": neg_prompt_embeds,
                "pos_prompt_embeds": pos_prompt_embeds,
                "neg_pooled_embeds": neg_pooled_embeds,
                "pos_pooled_embeds": pos_pooled_embeds,
                "add_time_ids_single": add_time_ids_single,
            }
            return self.cond

        text_embeddings = self.encode_text(
            text_prompt, negative_prompt=negative_prompt
        )  # neg, pos
        
        neg, pos = text_embeddings.chunk(2)

        self.cond = {
            "neg": neg,
            "pos": pos, 
        }

        return self.cond

    
    def init_latent(self, batch_size, latents=None):
        num_channels_latents = self.pipeline.unet.config.in_channels

        latents = self.pipeline.prepare_latents(
            batch_size,
            num_channels_latents,
            self.cfg.height,
            self.cfg.width,
            self.pipeline.dtype,
            self.pipeline.device,
            generator = None,
            latents = latents,
        )

        return latents

    def decode_latent(self, latent):
        if self.cfg.sd_model != "sdxl":
            return super().decode_latent(latent)

        vae = self.pipeline.vae
        flag = False
        if latent.dim() == 3:
            flag = True
            latent = latent.unsqueeze(0)

        latents = latent.to(vae.dtype)
        vae_orig_dtype = vae.dtype
        needs_upcasting = bool(getattr(vae.config, "force_upcast", False) and vae.dtype == torch.float16)

        if needs_upcasting:
            self.pipeline.upcast_vae()
            latents = latents.to(next(iter(vae.post_quant_conv.parameters())).dtype)
        elif latents.dtype != vae.dtype and torch.backends.mps.is_available():
            # Align dtype on MPS to avoid known backend misbehavior.
            self.pipeline.vae = vae.to(latents.dtype)
            vae = self.pipeline.vae

        has_latents_mean = hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None
        has_latents_std = hasattr(vae.config, "latents_std") and vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            channels = latents.shape[1]
            latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, channels, 1, 1)
            latents_std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, channels, 1, 1)
            latents = latents * latents_std / vae.config.scaling_factor + latents_mean
        else:
            latents = latents / vae.config.scaling_factor

        image = vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)

        if needs_upcasting:
            self.pipeline.vae.to(dtype=vae_orig_dtype)

        if flag:
            image = image.squeeze(0)
        return image.to(torch.float32)
    



    def predict(
        self, 
        x_t, 
        timestep, 
        guidance_scale=None, 
        return_dict=False, 
        text_prompt=None, 
        negative_prompt=None,
    ):

        # Predict the noise using the UNet model
        if x_t.shape[1] == 3:
            x_t = self.encode_image(x_t)

        self.prepare_cond(text_prompt, negative_prompt)
        
        noise_pred = []
        for _i in range(0, len(x_t), self.cfg.minibatch_size):
            cur_batch_size = min(self.cfg.minibatch_size, len(x_t) - _i)
            cur_x_t_batch = x_t[_i:_i+cur_batch_size]
            cur_t = timestep[_i:_i+cur_batch_size].view(-1)

            cfg_cur_x_t_batch = torch.cat([cur_x_t_batch] * 2)
            cfg_cur_t = torch.cat([cur_t] * 2)

            if self.cfg.sd_model == "sdxl":
                # Build [neg-block, pos-block] layout that matches
                # cfg_cur_x_t_batch = cat([cur_x_t_batch] * 2) = [x1..xN, x1..xN].
                # The legacy `repeat(cur_batch_size, 1, 1)` on a [neg, pos] tensor
                # produced an interleaved [neg, pos, neg, pos, ...] layout, which
                # collapsed CFG (text - uncond -> 0) and yielded noise on SDXL.
                neg_p = self.cond["neg_prompt_embeds"].repeat(cur_batch_size, 1, 1)
                pos_p = self.cond["pos_prompt_embeds"].repeat(cur_batch_size, 1, 1)
                cur_prompt_embeds = torch.cat([neg_p, pos_p], dim=0)

                neg_t = self.cond["neg_pooled_embeds"].repeat(cur_batch_size, 1)
                pos_t = self.cond["pos_pooled_embeds"].repeat(cur_batch_size, 1)
                cur_add_text_embeds = torch.cat([neg_t, pos_t], dim=0)

                add_t_block = self.cond["add_time_ids_single"].repeat(cur_batch_size, 1)
                cur_add_time_ids = torch.cat([add_t_block, add_t_block], dim=0)

                cur_noise_pred = self.pipeline.unet(
                    cfg_cur_x_t_batch,
                    cfg_cur_t,
                    encoder_hidden_states=cur_prompt_embeds,
                    timestep_cond=None,
                    cross_attention_kwargs=None,
                    added_cond_kwargs={
                        "text_embeds": cur_add_text_embeds,
                        "time_ids": cur_add_time_ids,
                    },
                    return_dict=False,
                )[0]
            else:
                cur_cond = {}
                for k, v in self.cond.items():
                    # neg, pos embeddings
                    cur_cond[k] = v.repeat(cur_batch_size, *([1] * (v.dim() - 1)))

                cur_cond["encoder_hidden_states"] = torch.cat([cur_cond["neg"], cur_cond["pos"]], dim=0)
                cur_cond.pop("neg", None)
                cur_cond.pop("pos", None)

                cur_noise_pred = self.pipeline.unet(
                    cfg_cur_x_t_batch,
                    cfg_cur_t,
                    timestep_cond=None,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=None,
                    return_dict=False,
                    **cur_cond,
                )[0]

            cur_noise_pred_uncond, cur_noise_pred_text = cur_noise_pred.chunk(2)
            cur_noise_pred = cur_noise_pred_uncond + self.cfg.guidance_scale * (
                cur_noise_pred_text - cur_noise_pred_uncond
            )

            if _sd_debug_enabled():
                with torch.no_grad():
                    diff = (cur_noise_pred_text - cur_noise_pred_uncond).float()
                    diff_norm_per_sample = diff.flatten(1).norm(dim=1).mean().item()
                    uncond_norm = cur_noise_pred_uncond.float().flatten(1).norm(dim=1).mean().item()
                    text_norm = cur_noise_pred_text.float().flatten(1).norm(dim=1).mean().item()
                _sd_debug_log(
                    f"predict sd_model={self.cfg.sd_model} batch={cur_batch_size} "
                    f"t_first={int(cur_t[0].item()) if cur_t.numel() > 0 else -1} "
                    f"uncond_norm={uncond_norm:.4f} text_norm={text_norm:.4f} "
                    f"text_minus_uncond_norm={diff_norm_per_sample:.4f}"
                )

            self.nfe += len(cfg_cur_x_t_batch)
            noise_pred.append(cur_noise_pred)

        noise_pred = torch.cat(noise_pred, dim=0)

        if return_dict:
            return {
                "noise_pred": noise_pred,
            }
        return noise_pred
    


    def get_variance(self, alpha_prod_t, alpha_prod_t_prev):
        # alpha_prod_t = self.alphas_cumprod[timestep]
        # alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance


    def step(
        self,
        x,
        t_curr,
        d_t,
        model_pred=None,
        prev_timestep=None,
    ):
        # Manual DDIM step (eta-stochastic) that uses BOTH the rbf-side t_curr
        # AND prev_timestep (each snapped to the nearest scheduler timestep)
        # for alpha indexing, instead of delegating to scheduler.step which
        # ignores `prev_timestep` and recomputes its own as
        # `timestep - num_train // num_inference`. That recomputation breaks
        # the rbf trajectory, since:
        #   - rbf t_curr step is t_max/max_steps (e.g. 999/64 ~= 15.6)
        #   - scheduler step is num_train//num_inference (e.g. 1000//64 = 15)
        # so multiple consecutive rbf t_curr values snap to the same scheduler
        # timestep but scheduler.step still advances the latent each call,
        # causing the latent's actual noise level to drift away from what the
        # rbf framework (and predict()) believe it to be.
        assert model_pred is not None, "model_pred must be provided"
        device = x.device
        snapped_t = self._snap_to_scheduler_timesteps(t_curr, device)

        if prev_timestep is None:
            # Fall back to scheduler-style next step.
            step_ratio = max(
                int(self.scheduler.config.num_train_timesteps)
                // max(int(self.scheduler.num_inference_steps or 1), 1),
                1,
            )
            snapped_prev = (snapped_t - step_ratio).clamp_(min=-1)
        else:
            snapped_prev = self._snap_to_scheduler_timesteps(prev_timestep, device)

        eta = float(self.cfg.eta)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        max_idx = int(alphas_cumprod.shape[0] - 1)
        final_alpha = (
            torch.tensor(1.0, device=device, dtype=alphas_cumprod.dtype)
            if bool(getattr(self.scheduler.config, "set_alpha_to_one", False))
            else alphas_cumprod[0]
        )

        prev_latents: List[torch.Tensor] = []
        for i in range(len(x)):
            t_i = int(snapped_t[i].item())
            tp_i = int(snapped_prev[i].item())
            x_i = x[i:i + 1]
            eps_i = model_pred[i:i + 1]

            t_i_c = max(0, min(max_idx, t_i))
            a_t = alphas_cumprod[t_i_c].to(x_i.dtype)
            if tp_i < 0:
                a_p = final_alpha.to(x_i.dtype)
            else:
                tp_i_c = max(0, min(max_idx, tp_i))
                a_p = alphas_cumprod[tp_i_c].to(x_i.dtype)

            if tp_i >= t_i:
                # Degenerate snap (e.g. early iters where rbf t_curr and
                # rbf prev_timestep both round to scheduler.timesteps[0]).
                # No-op step keeps the latent at its current noise level
                # rather than corrupting it with a wrong-direction update.
                _sd_debug_log(
                    f"step NOOP sd_model={self.cfg.sd_model} i={i} "
                    f"t_rbf={float(t_curr.reshape(-1)[i].item()):.4f} "
                    f"prev_rbf={float(prev_timestep.reshape(-1)[i].item()) if prev_timestep is not None else float('nan'):.4f} "
                    f"snap_t={t_i} snap_prev={tp_i} a_t={float(a_t.float().mean().item()):.6f} "
                    f"a_p={float(a_p.float().mean().item()):.6f}"
                )
                prev_latents.append(x_i)
                continue

            sqrt_a_t = torch.sqrt(a_t.clamp(min=1e-12))
            sqrt_a_p = torch.sqrt(a_p.clamp(min=1e-12))
            beta_t = (1.0 - a_t).clamp(min=0.0)
            beta_p = (1.0 - a_p).clamp(min=0.0)

            pred_x0 = (x_i - torch.sqrt(beta_t) * eps_i) / sqrt_a_t

            # DDIM stochastic variance term (eta-controlled).
            ratio = (a_t / a_p.clamp(min=1e-12)).clamp(max=1.0)
            sigma_t = (
                eta * torch.sqrt(beta_p / beta_t.clamp(min=1e-12)) * torch.sqrt((1.0 - ratio).clamp(min=0.0))
            )

            pred_dir = torch.sqrt(torch.clamp(beta_p - sigma_t * sigma_t, min=0.0)) * eps_i
            x_prev = sqrt_a_p * pred_x0 + pred_dir
            if eta > 0.0:
                noise = torch.randn_like(x_i)
                x_prev = x_prev + sigma_t * noise

            if _sd_debug_enabled():
                _sd_debug_log(
                    f"step OK sd_model={self.cfg.sd_model} i={i} "
                    f"t_rbf={float(t_curr.reshape(-1)[i].item()):.4f} "
                    f"prev_rbf={float(prev_timestep.reshape(-1)[i].item()) if prev_timestep is not None else float('nan'):.4f} "
                    f"snap_t={t_i} snap_prev={tp_i} "
                    f"a_t={float(a_t.float().mean().item()):.6f} a_p={float(a_p.float().mean().item()):.6f} "
                    f"x_norm={float(x_i.float().flatten().norm().item()):.4f} "
                    f"x_prev_norm={float(x_prev.float().flatten().norm().item()):.4f} "
                    f"x0_norm={float(pred_x0.float().flatten().norm().item()):.4f}"
                )

            prev_latents.append(x_prev)

        return torch.cat(prev_latents, dim=0)

    @staticmethod
    def _expand_like(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = value
        while out.ndim < target.ndim:
            out = out.unsqueeze(-1)
        return out
    

    def compute_velocity_transform_scheduler(self, x, t, **extras):
        timestep = self._snap_to_scheduler_timesteps(t, x.device)
        u_r = self.predict(x, timestep)

        return u_r

    def _snap_to_scheduler_timesteps(self, t: torch.Tensor, device: torch.device) -> torch.Tensor:
        query = t.reshape(-1).to(device=device, dtype=torch.float32)
        scheduler_timesteps = self.scheduler.timesteps.to(device=device, dtype=torch.float32)
        dists = (query[:, None] - scheduler_timesteps[None, :]).abs()
        nearest_idx = dists.argmin(dim=1)
        return scheduler_timesteps[nearest_idx].to(dtype=torch.long)

    def _alpha_like(self, t, target):
        # Snap continuous rbf-side `t` to the nearest scheduler timestep BEFORE
        # indexing `alphas_cumprod`. The eps prediction (computed via
        # compute_velocity_transform_scheduler -> predict) is produced at this
        # snapped timestep, so get_tweedie / get_eps must use the same snapped
        # alpha to be self-consistent. Without this, e.g. rbf t=999 would index
        # alphas_cumprod[999] (~0.005) while eps was predicted at scheduler t=945,
        # producing wildly mis-scaled tweedie estimates that the corrector
        # (and the saved cur_step_best_tweedie) treat as the final image.
        device = self.scheduler.alphas_cumprod.device
        snapped = self._snap_to_scheduler_timesteps(
            t.to(device=device, dtype=torch.float32), device
        )
        max_idx = int(self.scheduler.alphas_cumprod.shape[0] - 1)
        snapped = snapped.clamp_(min=0, max=max_idx)
        alpha = self.scheduler.alphas_cumprod[snapped].to(target)
        while alpha.ndim < target.ndim:
            alpha = alpha.unsqueeze(-1)
        return alpha
    

    # def get_tweedie(self, noisy_sample, model_pred, t):
    #     timestep = (t).to(model_pred)

    #     r_scheduler_output = cur_scheduler(t=timestep.float())

    #     alpha_r = r_scheduler_output.alpha_t.to(model_pred.dtype)
    #     sigma_r = r_scheduler_output.sigma_t.to(model_pred.dtype)
    #     d_alpha_r = r_scheduler_output.d_alpha_t.to(model_pred.dtype)
    #     d_sigma_r = r_scheduler_output.d_sigma_t.to(model_pred.dtype)

    #     numer = (sigma_r * model_pred) - (d_sigma_r * noisy_sample)
    #     denom = (d_alpha_r * sigma_r) - (d_sigma_r * alpha_r)

    #     return numer / denom


    
    def tau_func(
        self, 
        t_curr, 
        d_t,
    ):
        tau = (t_curr / 1000.0) * (d_t * self.cfg.tau_norm)
        # assert tau >= 0, f"Invalid tau value {tau}"

        return tau 


    def sample(self, text_prompt=None):
        if text_prompt is None:
            text_prompt = self.cfg.text_prompt

        self.prepare_cond()
        with torch.no_grad():
            images = self.pipeline(
                [text_prompt], negative_prompt=[self.cfg.negative_prompt]
            ).images
        return images
    
    def fast_sample(self, x_t, timesteps, guidance_scale=None, text_prompt=None, negative_prompt=None):
        self.fast_scheduler.set_timesteps(timesteps=timesteps)
        for t in timesteps:
            noise_pred = self.predict(x_t, t, guidance_scale, text_prompt, negative_prompt)
            x_t = self.fast_scheduler.step(noise_pred, t, x_t, return_dict=False)[0]

        return x_t