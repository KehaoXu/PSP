from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline

import rbf.shared_modules as sm
from rbf.prior.base import NEGATIVE_PROMPT, Prior
from rbf.prior.flux import FlowMatchEulerDiscreteScheduler, retrieve_timesteps
from rbf.utils.extra_utils import ignore_kwargs
from rbf.utils.print_utils import print_info, print_note


class StableDiffusion35Prior(Prior):
    @ignore_kwargs
    @dataclass
    class Config:
        device: int = 0
        batch_size: int = 1
        minibatch_size: int = 2
        n_particles: int = 1
        model_name: str = "stabilityai/stable-diffusion-3.5-large"
        text_prompt: str = (
            "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
        )
        negative_prompt: str = NEGATIVE_PROMPT
        width: int = 1024
        height: int = 1024
        guidance_scale: float = 7.0
        root_dir: str = "./results/default"
        max_steps: int = 32

        sample_method: str = "ode"
        diffusion_coefficient: str = "sigma"
        diffusion_norm: float = 1.0
        convert_scheduler: Optional[str] = None
        scheduler_n: Optional[float] = None
        t_max: float = 1000.0

        disable_debug: bool = False
        log_interval: int = 5

        exp_diff_coeff_sigma: float = 0.1

    def __init__(self, cfg):
        super().__init__()
        self.cfg = self.Config(**cfg)
        self._debug_step = 0

        if self.cfg.sample_method == "sde":
            assert self.cfg.diffusion_coefficient is not None
            assert self.cfg.diffusion_norm is not None

        print_info("Using prior model: ", self.cfg.model_name) if not sm.OFF_LOG else None
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.cfg.model_name,
            torch_dtype=torch.bfloat16,
        ).to(self.cfg.device)
        self.pipeline._guidance_scale = float(self.cfg.guidance_scale)

        self.pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config
        )

        self.nfe = 0

        sigmas = np.linspace(1.0, 1 / self.cfg.max_steps, self.cfg.max_steps)
        retrieve_timesteps(
            self.pipeline.scheduler,
            self.cfg.max_steps,
            self.pipeline.device,
            None,
            sigmas,
        )

        from rbf.prior.denoise_schedulers import CondOTScheduler

        self.original_scheduler = CondOTScheduler(device=self.pipeline.device)
        self.new_scheduler = None

        print(f"[*] Original scheduler set to {self.original_scheduler.__class__.__name__}") if not sm.OFF_LOG else None
        if self.cfg.convert_scheduler == "vp":
            from rbf.prior.denoise_schedulers import VPScheduler

            self.new_scheduler = VPScheduler(device=self.pipeline.device)
        elif self.cfg.convert_scheduler == "polynomial":
            from rbf.prior.denoise_schedulers import PolynomialConvexScheduler

            self.new_scheduler = PolynomialConvexScheduler(
                n=self.cfg.scheduler_n,
                device=self.pipeline.device,
            )
        elif self.cfg.convert_scheduler is None:
            print("[***] Not using scheduler conversion") if not sm.OFF_LOG else None
        else:
            raise NotImplementedError(
                f"convert_scheduler={self.cfg.convert_scheduler} is not supported for sd35 prior"
            )

        scheduler_name = (
            self.new_scheduler.__class__.__name__
            if self.new_scheduler is not None
            else self.original_scheduler.__class__.__name__
        )
        print_note(f"[***] Using scheduler conversion to {scheduler_name}") if not sm.OFF_LOG else None

    @property
    def rgb_res(self):
        return 1, 3, self.cfg.height, self.cfg.width

    @property
    def latent_res(self):
        latent_h = self.cfg.height // self.pipeline.vae_scale_factor
        latent_w = self.cfg.width // self.pipeline.vae_scale_factor
        return 1, self.pipeline.transformer.config.in_channels, latent_h, latent_w

    def prepare_cond(self, text_prompt=None, negative_prompt=None, _pass=False):
        if not _pass and hasattr(self, "cond"):
            return self.cond

        text_prompt = text_prompt if text_prompt is not None else self.cfg.text_prompt
        negative_prompt = (
            negative_prompt if negative_prompt is not None else self.cfg.negative_prompt
        )

        try:
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=self.pipeline.transformer.dtype,
                enabled=torch.cuda.is_available(),
            ):
                pe, ne, ppe, npe = self.pipeline.encode_prompt(
                    prompt=[text_prompt],
                    prompt_2=None,
                    prompt_3=None,
                    negative_prompt=[negative_prompt],
                    negative_prompt_2=None,
                    negative_prompt_3=None,
                    do_classifier_free_guidance=True,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    pooled_prompt_embeds=None,
                    negative_pooled_prompt_embeds=None,
                    device=self.pipeline.device,
                    clip_skip=None,
                    num_images_per_prompt=1,
                    max_sequence_length=256,
                    lora_scale=None,
                )
        except RuntimeError as exc:
            # Some environments run fp16 autocast by default while SD3.5 checkpoints are bf16.
            # If text projection sees fp16 activations with bf16 weights, retry after coercing
            # SD3 text encoders to fp16.
            if "expected mat1 and mat2 to have the same dtype" not in str(exc):
                raise
            print_note("[SD35 transfer] prompt encoding dtype mismatch detected; retrying with fp16 text encoders") if not sm.OFF_LOG else None
            self._coerce_text_encoders_dtype(torch.float16)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=torch.cuda.is_available(),
            ):
                pe, ne, ppe, npe = self.pipeline.encode_prompt(
                    prompt=[text_prompt],
                    prompt_2=None,
                    prompt_3=None,
                    negative_prompt=[negative_prompt],
                    negative_prompt_2=None,
                    negative_prompt_3=None,
                    do_classifier_free_guidance=True,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    pooled_prompt_embeds=None,
                    negative_pooled_prompt_embeds=None,
                    device=self.pipeline.device,
                    clip_skip=None,
                    num_images_per_prompt=1,
                    max_sequence_length=256,
                    lora_scale=None,
                )

        self.cond = {
            "positive_prompt_embeds": pe,
            "negative_prompt_embeds": ne,
            "positive_pooled_prompt_embeds": ppe,
            "negative_pooled_prompt_embeds": npe,
        }
        return self.cond

    def _coerce_text_encoders_dtype(self, target_dtype: torch.dtype) -> None:
        for module_name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            module = getattr(self.pipeline, module_name, None)
            if module is not None:
                module.to(dtype=target_dtype)

    def sample(self, text_prompt=None):
        if text_prompt is None:
            text_prompt = self.cfg.text_prompt

        with torch.no_grad():
            images = self.pipeline(prompt=text_prompt, output_type="latent")
        return images

    def init_latent(self, batch_size, latents=None):
        latents = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=self.pipeline.transformer.config.in_channels,
            height=self.cfg.height,
            width=self.cfg.width,
            dtype=self.pipeline.transformer.dtype,
            device=self.pipeline.device,
            generator=None,
            latents=latents,
        )
        # FITS optimizers wrap this tensor into nn.Parameter.
        # Ensure it is not an inference-mode tensor.
        return latents.detach().clone()

    def predict(
        self,
        x_t,
        timestep,
        return_dict=False,
        text_prompt=None,
        negative_prompt=None,
    ):
        if x_t.shape[1] == 3:
            x_t = self.encode_image(x_t)

        cond = self.prepare_cond(text_prompt, negative_prompt)

        t = timestep.to(x_t).reshape(-1)
        if t.numel() == 1:
            t = t.repeat(x_t.shape[0])

        noise_pred = []
        for i in range(0, len(x_t), self.cfg.minibatch_size):
            cur_batch_size = min(self.cfg.minibatch_size, len(x_t) - i)
            cur_x_t_batch = x_t[i : i + cur_batch_size]
            cur_t = t[i : i + cur_batch_size]

            pos_prompt = cond["positive_prompt_embeds"].repeat(cur_batch_size, 1, 1).to(cur_x_t_batch.dtype)
            neg_prompt = cond["negative_prompt_embeds"].repeat(cur_batch_size, 1, 1).to(cur_x_t_batch.dtype)
            pos_pooled = cond["positive_pooled_prompt_embeds"].repeat(cur_batch_size, 1).to(cur_x_t_batch.dtype)
            neg_pooled = cond["negative_pooled_prompt_embeds"].repeat(cur_batch_size, 1).to(cur_x_t_batch.dtype)

            latent_input = torch.cat([cur_x_t_batch, cur_x_t_batch], dim=0)
            prompt_embeds = torch.cat([neg_prompt, pos_prompt], dim=0)
            pooled_prompt_embeds = torch.cat([neg_pooled, pos_pooled], dim=0)
            model_t = torch.cat([cur_t, cur_t], dim=0)

            cur_noise_pred = self.pipeline.transformer(
                hidden_states=latent_input,
                timestep=model_t,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=getattr(self.pipeline, "joint_attention_kwargs", None),
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = cur_noise_pred.chunk(2)
            cur_noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            self.nfe += cur_batch_size
            noise_pred.append(cur_noise_pred)

        noise_pred = torch.cat(noise_pred, dim=0)
        if return_dict:
            return {"noise_pred": noise_pred}
        return noise_pred

    def encode_image(self, img_tensor):
        assert self.pipeline is not None, "Pipeline not initialized"
        vae = self.pipeline.vae
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        x = (2 * img_tensor - 1).to(vae.dtype)
        latent = vae.encode(x).latent_dist.sample()
        if getattr(vae.config, "shift_factor", None) is not None:
            latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
        else:
            latent = latent * vae.config.scaling_factor
        return latent

    def decode_latent(self, latent, convert_to_float=True):
        assert self.pipeline is not None, "Pipeline not initialized"
        vae = self.pipeline.vae
        flag = False
        if latent.dim() == 3:
            flag = True
            latent = latent.unsqueeze(0)

        needs_upcast = vae.dtype == torch.float16 and getattr(vae.config, "force_upcast", False)
        if needs_upcast:
            if hasattr(self.pipeline, "upcast_vae"):
                self.pipeline.upcast_vae()
            else:
                vae.to(dtype=torch.float32)

        if getattr(vae, "post_quant_conv", None) is not None:
            decode_param = next(vae.post_quant_conv.parameters())
        else:
            decode_param = next(vae.parameters())

        latent = (latent / vae.config.scaling_factor).to(
            device=decode_param.device,
            dtype=decode_param.dtype,
        )
        if getattr(vae.config, "shift_factor", None) is not None:
            latent = latent + vae.config.shift_factor

        image = vae.decode(latent, return_dict=False)[0]

        if needs_upcast:
            vae.to(dtype=torch.float16)

        image = (image / 2 + 0.5).clamp(0, 1)
        if flag:
            image = image.squeeze(0)

        if convert_to_float:
            return image.to(torch.float32)
        return image

    def decode_latent_if_needed(self, latent):
        if latent.shape[-3] == self.pipeline.transformer.config.in_channels:
            return self.decode_latent(latent)
        return latent

    def step(
        self,
        x,
        t_curr,
        d_t,
        model_pred=None,
        prev_timestep=None,
    ):
        if self.cfg.sample_method not in ["ode", "sde"]:
            raise NotImplementedError(f"Unsupported sample method {self.cfg.sample_method}")
        assert model_pred is not None, "Model prediction not provided"
        assert torch.all(d_t >= 0.0).item() and torch.all(d_t <= 1.0).item(), f"Invalid time step {d_t}"

        t_curr_expanded = self._expand_like(t_curr.to(x), x)
        d_t_expanded = self._expand_like(d_t.to(x), x)

        diffuse = self.pipeline.scheduler.get_diffuse(
            t_curr_expanded,
            self.cfg.sample_method,
            self.cfg.diffusion_coefficient,
            self.cfg.diffusion_norm,
            self.cfg.convert_scheduler,
            new_scheduler=self.new_scheduler,
            original_scheduler=self.original_scheduler,
        )

        drift = self.pipeline.scheduler.get_drift(
            x,
            model_pred,
            t_curr_expanded,
            self.cfg.sample_method,
            diffusion_coefficient=self.cfg.diffusion_coefficient,
            diffusion_norm=self.cfg.diffusion_norm,
            convert_scheduler=self.cfg.convert_scheduler,
            new_scheduler=self.new_scheduler,
            original_scheduler=self.original_scheduler,
            diffuse=diffuse,
        )

        prev_x_mean = x + drift * d_t_expanded
        w = torch.randn(x.size()).to(x)
        dw = w * torch.sqrt(torch.abs(d_t_expanded))
        prev_latent = prev_x_mean + diffuse * dw

        self._debug_step += 1
        if (
            not self.cfg.disable_debug
            and not sm.OFF_LOG
            and self._debug_step % max(int(self.cfg.log_interval), 1) == 0
        ):
            sigma = self._current_sigma(t_curr)
            tweedie = self.get_tweedie(x, model_pred, t_curr)
            print_note(
                "[SD35 transfer]",
                f"step={self._debug_step}",
                f"sigma_mean={sigma.mean().item():.5f}",
                f"diffuse_mean={diffuse.float().mean().item():.5f}",
                f"drift_norm={drift.float().norm().item():.5f}",
                f"vel_norm={model_pred.float().norm().item():.5f}",
                f"x0_norm={tweedie.float().norm().item():.5f}",
            )

        return prev_latent

    def _current_sigma(self, t):
        # FITS SDE logic uses continuous time (t/1000) in scheduler-conversion space.
        # For diagnostics we should report the same continuous sigma, not a discretized
        # index from pipeline.scheduler.timesteps.
        t = t.to(device=self.pipeline.device, dtype=torch.float32)
        t_cont = self._expand_like(t / 1000.0, t)
        cur_scheduler = self.new_scheduler if self.cfg.convert_scheduler is not None else self.original_scheduler
        sigma = cur_scheduler(t=t_cont).sigma_t.to(device=self.pipeline.device, dtype=torch.float32)
        return sigma.reshape(-1)

    def get_tweedie(self, noisy_sample, model_pred, t):
        timestep = self._expand_like((t / 1000.0).to(model_pred), noisy_sample)
        if self.cfg.convert_scheduler is not None:
            print_note("[***] Using converted scheduler in computing Tweedies") if not sm.OFF_LOG else None
            cur_scheduler = self.new_scheduler
        else:
            cur_scheduler = self.original_scheduler

        r_scheduler_output = cur_scheduler(t=timestep.float())
        alpha_r = self._expand_like(r_scheduler_output.alpha_t.to(model_pred.dtype), noisy_sample)
        sigma_r = self._expand_like(r_scheduler_output.sigma_t.to(model_pred.dtype), noisy_sample)
        d_alpha_r = self._expand_like(r_scheduler_output.d_alpha_t.to(model_pred.dtype), noisy_sample)
        d_sigma_r = self._expand_like(r_scheduler_output.d_sigma_t.to(model_pred.dtype), noisy_sample)

        numer = (sigma_r * model_pred) - (d_sigma_r * noisy_sample)
        denom = (d_alpha_r * sigma_r) - (d_sigma_r * alpha_r)
        return numer / denom

    def compute_velocity_transform_scheduler(self, x, t, **extras):
        t = self._expand_like((t / 1000.0).to(x).to(torch.float32), x)
        r = t.clone()

        conversion_scheduler = self.new_scheduler if self.new_scheduler is not None else self.original_scheduler
        r_scheduler_output = conversion_scheduler(t=r)

        alpha_r = self._expand_like(r_scheduler_output.alpha_t, x)
        sigma_r = self._expand_like(r_scheduler_output.sigma_t, x)
        d_alpha_r = self._expand_like(r_scheduler_output.d_alpha_t, x)
        d_sigma_r = self._expand_like(r_scheduler_output.d_sigma_t, x)

        t = self.original_scheduler.snr_inverse(alpha_r / sigma_r)
        t_scheduler_output = self.original_scheduler(t=t)

        alpha_t = self._expand_like(t_scheduler_output.alpha_t, x)
        sigma_t = self._expand_like(t_scheduler_output.sigma_t, x)
        d_alpha_t = self._expand_like(t_scheduler_output.d_alpha_t, x)
        d_sigma_t = self._expand_like(t_scheduler_output.d_sigma_t, x)

        s_r = sigma_r / sigma_t
        dt_r = (
            sigma_t
            * sigma_t
            * (sigma_r * d_alpha_r - alpha_r * d_sigma_r)
            / (sigma_r * sigma_r * (sigma_t * d_alpha_t - alpha_t * d_sigma_t))
        )
        ds_r = (sigma_t * d_sigma_r - sigma_r * d_sigma_t * dt_r) / (sigma_t * sigma_t)

        u_t = self.predict((x / s_r).to(x.dtype), (t * 1000.0).to(x.dtype))
        u_r = (ds_r * x / s_r + dt_r * s_r * u_t).to(x.dtype)
        return u_r

    @staticmethod
    def _expand_like(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = value
        while out.ndim < target.ndim:
            out = out.unsqueeze(-1)
        return out

    @property
    def device(self):
        return self.pipeline.device

    @property
    def dtype(self):
        return self.pipeline.vae.dtype
