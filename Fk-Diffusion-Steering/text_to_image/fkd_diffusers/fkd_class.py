"""
Feynman-Kac Diffusion (FKD) steering mechanism implementation.
"""

import json
import os
import time
import torch
from enum import Enum
import numpy as np
from typing import Callable, Optional, Tuple
import logging
from smc_utils import resampling_function

_DEBUG_ENABLED = os.environ.get("FK_DEBUG_LOG", "").lower() in {"1", "true", "yes", "on"}
_DEBUG_LOG_PATH = os.environ.get("FK_DEBUG_LOG_PATH", ".cursor/debug.log")


def _debug_log(*, location, message, data, hypothesis_id):
    if not _DEBUG_ENABLED:
        return
    payload = {
        "sessionId": "debug-session",
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    debug_dir = os.path.dirname(_DEBUG_LOG_PATH)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload) + "\n")


class PotentialType(Enum):
    DIFF = "diff"
    MAX = "max"
    ADD = "add"
    RT = "rt"


class FKD:
    """
    Implements the FKD steering mechanism. Should be initialized along the diffusion process. .resample() should be invoked at each diffusion timestep.
    See FKD fkd_pipeline_sdxl
    Args:
        potential_type: Type of potential function must be one of PotentialType.
        lmbda: Lambda hyperparameter controlling weight scaling.
        num_particles: Number of particles to maintain in the population.
        adaptive_resampling: Whether to perform adaptive resampling.
        resample_frequency: Frequency (in timesteps) to perform resampling.
        resampling_t_start: Timestep to start resampling.
        resampling_t_end: Timestep to stop resampling.
        time_steps: Total number of timesteps in the sampling process.
        reward_fn: Function to compute rewards from decoded latents.
        reward_min_value: Minimum value for rewards (default: 0.0). Important for the Max potential type.
        latent_to_decode_fn: Function to decode latents to images, relevant for latent diffusion models (default: identity function).
        device: Device on which computations will be performed (default: CUDA).
        **kwargs: Additional keyword arguments, unused.
    """

    def __init__(
        self,
        *,
        potential_type: PotentialType,
        lmbda: float,
        num_particles: int,
        adaptive_resampling: bool,
        resample_frequency: int,
        resampling_t_start: int,
        resampling_t_end: int,
        time_steps: int,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        reward_min_value: float = 0.0,
        latent_to_decode_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        device: torch.device = torch.device('cuda'),
        log_reward_every_step: bool = False,
        resampling: str = "multinomial",
        tempering_schedule: str = "constant",
        **kwargs,
    ) -> None:
        # Initialize hyperparameters and functions

        # if kwargs:
            # logging.warning(f"FKD Steering - Unused arguments: {kwargs}")

        self.potential_type = PotentialType(potential_type)
        self.lmbda = lmbda
        self.num_particles = num_particles
        self.adaptive_resampling = adaptive_resampling
        self.resample_frequency = resample_frequency
        self.resampling_t_start = resampling_t_start
        self.resampling_t_end = resampling_t_end
        self.time_steps = time_steps

        self.reward_fn = reward_fn
        self.latent_to_decode_fn = latent_to_decode_fn
        self.log_reward_every_step = log_reward_every_step
        self.debug_resampling = bool(kwargs.get("debug_resampling", False))
        # Optional plotting/debug mode: do not force a resample at the final step.
        self.disable_forced_final_resampling = bool(
            kwargs.get("disable_forced_final_resampling", False)
        )

        # Initialize device and population reward state
        self.device = device
        self.resample_fn = resampling_function(resample_strategy=resampling)
        self.tempering_schedule = tempering_schedule

        # initial rewards
        self.population_rs = (
            torch.ones(self.num_particles, device=self.device) * reward_min_value
        )
        self.product_of_potentials = torch.ones(self.num_particles).to(self.device)
        self.trajectory_log = self._init_trajectory_log()
        self.trajectory_log_raw = self._init_trajectory_log()
        self.last_resample_info = {
            "sampling_idx": None,
            "did_resample": False,
            "parents": [i for i in range(self.num_particles)],
            "kill_mask": [0 for _ in range(self.num_particles)],
        }

    def _init_trajectory_log(self):
        return {
            "timesteps": [],
            "rewards": [],
            "kills": [],
            "parents": [],
            "num_particles": self.num_particles,
            "resampling_t_start": self.resampling_t_start,
            "resampling_t_end": self.resampling_t_end,
            "resample_frequency": self.resample_frequency,
            "log_reward_every_step": self.log_reward_every_step,
            "time_steps": self.time_steps,
        }

    def _log_step_in(self, *, log, sampling_idx: int, rewards_list, parents, kill_mask):
        if log["timesteps"] and log["timesteps"][-1] == int(sampling_idx):
            log["rewards"][-1] = rewards_list
            log["kills"][-1] = kill_mask
            log["parents"][-1] = parents
            return

        log["timesteps"].append(int(sampling_idx))
        log["rewards"].append(rewards_list)
        log["kills"].append(kill_mask)
        log["parents"].append(parents)

    def _update_step_meta(self, *, log, sampling_idx: int, parents, kill_mask):
        if not log["timesteps"]:
            return
        if log["timesteps"][-1] == int(sampling_idx):
            log["kills"][-1] = kill_mask
            log["parents"][-1] = parents
            return
        if int(sampling_idx) in log["timesteps"]:
            idx = log["timesteps"].index(int(sampling_idx))
            log["kills"][idx] = kill_mask
            log["parents"][idx] = parents

    def log_step(self, *, sampling_idx: int, rs_candidates: torch.Tensor) -> None:
        rewards_list = rs_candidates.detach().cpu().tolist()
        parents = [i for i in range(self.num_particles)]
        kill_mask = [0 for _ in range(self.num_particles)]
        self._log_step_in(
            log=self.trajectory_log_raw,
            sampling_idx=sampling_idx,
            rewards_list=rewards_list,
            parents=parents,
            kill_mask=kill_mask,
        )

    def resample(
        self, *, sampling_idx: int, latents: torch.Tensor, x0_preds: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform resampling of particles if conditions are met.
        Should be invoked at each timestep in the reverse diffusion process.

        Args:
            sampling_idx: Current sampling index (timestep).
            latents: Current noisy latents.
            x0_preds: Predictions for x0 based on latents.

        Returns:
            A tuple containing resampled latents and optionally resampled images.
        """
        # Check if resampling is within the allowed range and conditions
        resampling_interval = np.arange(
            self.resampling_t_start, self.resampling_t_end + 1, self.resample_frequency
        )
        if not self.disable_forced_final_resampling:
            resampling_interval = np.append(resampling_interval, self.time_steps - 1)

        _debug_log(
            location="fkd_class.py:resample",
            message="Resample check",
            data={
                "sampling_idx": int(sampling_idx),
                "in_interval": bool(sampling_idx in resampling_interval),
                "resampling_t_start": int(self.resampling_t_start),
                "resampling_t_end": int(self.resampling_t_end),
                "resample_frequency": int(self.resample_frequency),
            },
            hypothesis_id="H1",
        )

        if sampling_idx not in resampling_interval:
            self.last_resample_info = {
                "sampling_idx": int(sampling_idx),
                "did_resample": False,
                "parents": [i for i in range(self.num_particles)],
                "kill_mask": [0 for _ in range(self.num_particles)],
            }
            return latents, None

        # Decode latents to population images and compute rewards
        population_images = self.latent_to_decode_fn(x0_preds)
        rs_candidates = self.reward_fn(population_images)
        if self.tempering_schedule == "increase":
            rs_candidates = rs_candidates * (sampling_idx / self.time_steps * 2)
        elif self.tempering_schedule == "constant":
            pass
        else:
            raise ValueError(f"Unknown tempering schedule: {self.tempering_schedule}")

        _debug_log(
            location="fkd_class.py:resample",
            message="Reward stats",
            data={
                "sampling_idx": int(sampling_idx),
                "min": float(torch.min(rs_candidates).item()),
                "max": float(torch.max(rs_candidates).item()),
                "mean": float(torch.mean(rs_candidates).item()),
                "std": float(torch.std(rs_candidates).item()),
            },
            hypothesis_id="H2",
        )

        # Compute importance weights
        if self.potential_type == PotentialType.MAX:
            rs_candidates = torch.max(rs_candidates, self.population_rs)
            w = torch.exp(self.lmbda * rs_candidates)
        elif self.potential_type == PotentialType.ADD:
            rs_candidates = rs_candidates + self.population_rs
            w = torch.exp(self.lmbda * rs_candidates)
        elif self.potential_type == PotentialType.DIFF:
            diffs = rs_candidates - self.population_rs
            w = torch.exp(self.lmbda * diffs)
        elif self.potential_type == PotentialType.RT:
            w = torch.exp(self.lmbda * rs_candidates)
        else:
            raise ValueError(f"potential_type {self.potential_type} not recognized")

        if sampling_idx == self.time_steps - 1:
            if (
                self.potential_type == PotentialType.MAX
                or self.potential_type == PotentialType.ADD
                or self.potential_type == PotentialType.RT
            ):
                w = torch.exp(self.lmbda * rs_candidates) / self.product_of_potentials

        w = torch.clamp(w, 0, 1e10)
        w[torch.isnan(w)] = 0.0
        weight_sum = w.sum()
        if weight_sum.item() > 0:
            normalized_w = w / weight_sum
        else:
            normalized_w = torch.full_like(w, 1.0 / self.num_particles)

        resampled = False
        indices = None
        if self.adaptive_resampling or (
            sampling_idx == self.time_steps - 1 and not self.disable_forced_final_resampling
        ):
            # compute effective sample size
            ess = 1.0 / (normalized_w.pow(2).sum())

            if ess < 0.5 * self.num_particles:
                print(f"Resampling at timestep {sampling_idx} with ESS: {ess}")
                # Resample indices based on weights
                indices = torch.multinomial(
                    w, num_samples=self.num_particles, replacement=True
                )
                resampled = True
                resampled_latents = latents[indices]
                self.population_rs = rs_candidates[indices]

                # Resample population images
                resampled_images = population_images[indices]

                # Update product of potentials; used for max and add potentials
                self.product_of_potentials = (
                    self.product_of_potentials[indices] * w[indices]
                )
            else:
                # No resampling
                resampled_images = population_images
                resampled_latents = latents
                self.population_rs = rs_candidates

        else:
            # Resample indices based on selected FK strategy (e.g. ssp, multinomial).
            log_w = torch.log(normalized_w.clamp(min=1e-20)).unsqueeze(0)
            indices, _, _ = self.resample_fn(log_w)
            indices = indices.squeeze(0).to(device=latents.device, dtype=torch.long)
            resampled = True
            resampled_latents = latents[indices]
            self.population_rs = rs_candidates[indices]

            # Resample population images
            resampled_images = population_images[indices]

            # Update product of potentials; used for max and add potentials
            self.product_of_potentials = (
                self.product_of_potentials[indices] * w[indices]
            )

        if self.debug_resampling:
            # Print reward/weight diagnostics at each configured resampling step.
            print(
                "[FKD debug] step="
                f"{int(sampling_idx)} rewards={rs_candidates.detach().cpu().tolist()} "
                f"normalized_weights={normalized_w.detach().cpu().tolist()}"
            )

        rewards_list = rs_candidates.detach().cpu().tolist()
        _debug_log(
            location="fkd_class.py:resample",
            message="Rewards list stats",
            data={
                "sampling_idx": int(sampling_idx),
                "rewards_min": float(min(rewards_list)) if rewards_list else None,
                "rewards_max": float(max(rewards_list)) if rewards_list else None,
                "rewards_sample": rewards_list[: min(3, len(rewards_list))],
            },
            hypothesis_id="H8",
        )
        if resampled and indices is not None:
            kill_mask = [
                int(indices[i].item() != i) for i in range(self.num_particles)
            ]
            parents = [int(idx.item()) for idx in indices]
        else:
            kill_mask = [0 for _ in range(self.num_particles)]
            parents = [i for i in range(self.num_particles)]

        if self.debug_resampling:
            print(
                "[FKD debug] step="
                f"{int(sampling_idx)} resampled={bool(resampled)} parents={parents} kill_mask={kill_mask}"
            )

        self.last_resample_info = {
            "sampling_idx": int(sampling_idx),
            "did_resample": bool(resampled),
            "parents": parents,
            "kill_mask": kill_mask,
        }

        self._log_step_in(
            log=self.trajectory_log,
            sampling_idx=sampling_idx,
            rewards_list=rewards_list,
            parents=parents,
            kill_mask=kill_mask,
        )
        if self.log_reward_every_step:
            self._update_step_meta(
                log=self.trajectory_log_raw,
                sampling_idx=sampling_idx,
                parents=parents,
                kill_mask=kill_mask,
            )

        _debug_log(
            location="fkd_class.py:resample",
            message="Trajectory append",
            data={
                "sampling_idx": int(sampling_idx),
                "resampled": bool(resampled),
                "kills": int(sum(kill_mask)),
                "log_len": len(self.trajectory_log["timesteps"]),
            },
            hypothesis_id="H3",
        )

        return resampled_latents, resampled_images


if __name__ == "__main__":

    # Demonstration of FKD resampling step
    import matplotlib.pyplot as plt
    import random

    # set seed
    random.seed(0)

    # 1x1 pixel images
    num_particles = 8
    x0s = torch.rand(num_particles, 1, 1)

    # reward darker images
    reward_function = lambda x: -0.5 * x.sum(dim=(1, 2))

    # Define the FKD steering mechanism
    fkds = FKD(
        potential_type=PotentialType.DIFF,
        lmbda=10.0,
        num_particles=num_particles,
        adaptive_resampling=False,
        resample_frequency=1,
        resampling_t_start=-1,
        resampling_t_end=100,
        time_steps=100,
        reward_fn=lambda x: reward_function(x),
        device=torch.device('cpu'),
    )

    # Define the sampling index
    sampling_idx = 0

    # Perform resampling
    resampled_latents, resampled_images = fkds.resample(
        sampling_idx=sampling_idx,
        latents=x0s,
        x0_preds=x0s,
    )

    plt.rc('text', usetex=True)
    fig, axs = plt.subplots(2, num_particles)

    axs[0, 0].set_title('Initial')
    axs[1, 0].set_title('Resampled')

    for i in range(num_particles):
        axs[0, i].imshow(x0s[i].detach().numpy(), cmap='gray', vmin=0, vmax=1)
        axs[1, i].imshow(
            resampled_images[i].detach().numpy(), cmap='gray', vmin=0, vmax=1
        )

        axs[1, i].axis('off')
        axs[0, i].axis('off')

    out_path = 'resampled_examples.png'
    plt.savefig(out_path)
    print('Saved resampled examples to:', out_path)
