"""
Utility functions for the FKD pipeline.
"""
import json
import os
import time
import torch
from diffusers import DDIMScheduler

_DEBUG_ENABLED = os.environ.get("FK_DEBUG_LOG", "").lower() in {"1", "true", "yes", "on"}
_DEBUG_LOG_PATH = os.environ.get("FK_DEBUG_LOG_PATH", ".cursor/debug.log")


def _debug_log(*, location, message, data, hypothesis_id):
    if not _DEBUG_ENABLED:
        return
    payload = {
        "sessionId": "debug-session",
        "runId": "post-fix",
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


# #region agent log
_debug_log(
    location="fks_utils.py:import",
    message="Import start",
    data={"file": __file__, "cwd": os.getcwd(), "sys_path_head": __import__("sys").path[:5]},
    hypothesis_id="H1",
)
# #endregion

try:
    # #region agent log
    _debug_log(
        location="fks_utils.py:import",
        message="Attempt import fkd_pipeline_sdxl",
        data={"dir_contents": os.listdir(os.path.dirname(__file__))},
        hypothesis_id="H1",
    )
    # #endregion
    from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL
except Exception as exc:
    # #region agent log
    _debug_log(
        location="fks_utils.py:import",
        message="Import fkd_pipeline_sdxl failed",
        data={"error": repr(exc)},
        hypothesis_id="H1",
    )
    # #endregion
    raise

try:
    # #region agent log
    _debug_log(
        location="fks_utils.py:import",
        message="Attempt import fkd_pipeline_sd",
        data={"dir_contents": os.listdir(os.path.dirname(__file__))},
        hypothesis_id="H2",
    )
    # #endregion
    from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion
except Exception as exc:
    # #region agent log
    _debug_log(
        location="fks_utils.py:import",
        message="Import fkd_pipeline_sd failed",
        data={"error": repr(exc)},
        hypothesis_id="H2",
    )
    # #endregion
    raise

from fkd_diffusers.rewards import (
    do_clip_score,
    do_clip_score_diversity,
    do_image_reward,
    do_human_preference_score,
    do_llm_grading
)


def get_model(model_name):
    """
    Get the FKD-supported model based on the model name.
    """
    if model_name == "stable-diffusion-xl":
        pipeline = FKDStableDiffusionXL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    elif model_name == "stable-diffusion-v1-5":
        pipeline = FKDStableDiffusion.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    elif model_name == "stable-diffusion-v1-4":
        pipeline = FKDStableDiffusion.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    elif model_name == "stable-diffusion-2-1":
        pipeline = FKDStableDiffusion.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    return pipeline



def do_eval(*, prompt, images, metrics_to_compute):
    """
    Compute the metrics for the given images and prompt.
    """
    results = {}
    for metric in metrics_to_compute:
        if metric == "Clip-Score":
            results[metric] = {}
            (
                results[metric]["result"],
                results[metric]["diversity"],
            ) = do_clip_score_diversity(images=images, prompts=prompt)
            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        elif metric == "ImageReward":
            results[metric] = {}
            results[metric]["result"] = do_image_reward(images=images, prompts=prompt)

            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        elif metric == "Clip-Score-only":
            results[metric] = {}
            results[metric]["result"] = do_clip_score(images=images, prompts=prompt)

            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()
        elif metric == "HumanPreference":
            results[metric] = {}
            results[metric]["result"] = do_human_preference_score(
                images=images, prompts=prompt
            )

            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        elif metric == "LLMGrader":
            results[metric] = {}
            out = do_llm_grading(images=images, prompts=prompt)
            print(out)
            results[metric]["result"] = out

            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results
