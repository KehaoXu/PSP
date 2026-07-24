import json
import os
import sys
import time
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


def _install_turtle_shim_for_hpsv2() -> None:
    """
    hpsv2's open_clip imports `from turtle import forward`, which can pull
    tkinter/X11 in headless environments and crash. Install a tiny shim so the
    import resolves without GUI dependencies.
    """
    enabled = os.environ.get("FK_HPS_TURTLE_SHIM", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return
    if "turtle" not in sys.modules:
        shim = types.ModuleType("turtle")
        shim.forward = lambda *_args, **_kwargs: None
        sys.modules["turtle"] = shim


_HPSV2 = None

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
    location="rewards.py:import",
    message="Import start",
    data={"file": __file__, "cwd": os.getcwd()},
    hypothesis_id="H7",
)
# #endregion

try:
    # #region agent log
    _debug_log(
        location="rewards.py:import",
        message="Attempt import .image_reward_utils and .llm_grading",
        data={"dir_contents": os.listdir(os.path.dirname(__file__))},
        hypothesis_id="H7",
    )
    # #endregion
    from .image_reward_utils import rm_load
    from .llm_grading import LLMGrader
except Exception as exc:
    # #region agent log
    _debug_log(
        location="rewards.py:import",
        message="Relative import failed",
        data={"error": repr(exc)},
        hypothesis_id="H7",
    )
    # #endregion
    raise

# Stores the reward models
REWARDS_DICT = {
    "Clip-Score": None,
    "ImageReward": None,
    "LLMGrader": None,
}


# Returns the reward function based on the guidance_reward_fn name
def get_reward_function(reward_name, images, prompts, metric_to_chase="overall_score"):
    if reward_name != "LLMGrader":
        print("`metric_to_chase` will be ignored as it only applies to 'LLMGrader' as the `reward_name`")
    if reward_name == "ImageReward":
        return do_image_reward(images=images, prompts=prompts)
    
    elif reward_name == "Clip-Score":
        return do_clip_score(images=images, prompts=prompts)
    
    elif reward_name == "HumanPreference":
        return do_human_preference_score(images=images, prompts=prompts)

    elif reward_name == "LLMGrader":
        return do_llm_grading(images=images, prompts=prompts, metric_to_chase=metric_to_chase)
    
    else:
        raise ValueError(f"Unknown metric: {reward_name}")
    
# Compute human preference score
def do_human_preference_score(*, images, prompts, use_paths=False):
    global _HPSV2
    if _HPSV2 is None:
        _install_turtle_shim_for_hpsv2()
        import hpsv2 as _hpsv2  # type: ignore

        _HPSV2 = _hpsv2

    if use_paths:
        scores = _HPSV2.score(images, prompts, hps_version="v2.1")
        scores = [float(score) for score in scores]
    else:
        scores = []
        for i, image in enumerate(images):
            score = _HPSV2.score(image, prompts[i], hps_version="v2.1")
            # print(f"Human preference score for image {i}: {score}")
            score = float(score[0])
            scores.append(score)

    # print(f"Human preference scores: {scores}")
    return scores

# Compute CLIP-Score and diversity
def do_clip_score_diversity(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["Clip-Score"] is None:
        REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=".", device="cuda")
    with torch.no_grad():
        arr_clip_result = []
        arr_img_features = []
        for i, prompt in enumerate(prompts):
            clip_result, feature_vect = REWARDS_DICT["Clip-Score"].score(
                prompt, images[i], return_feature=True
            )

            arr_clip_result.append(clip_result.item())
            arr_img_features.append(feature_vect['image'])

    # calculate diversity by computing pairwise similarity between image features
    diversity = torch.zeros(len(images), len(images))
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            diversity[i, j] = (arr_img_features[i] - arr_img_features[j]).pow(2).sum()
            diversity[j, i] = diversity[i, j]
    n_samples = len(images)
    diversity = diversity.sum() / (n_samples * (n_samples - 1))

    return arr_clip_result, diversity.item()

# Compute ImageReward
def do_image_reward(
    *,
    images=None,
    prompts,
    image_tensors=None,
    from_minus_one_to_one: bool = True,
    emulate_pil_uint8_roundtrip: bool = True,
    return_profile: bool = False,
):
    global REWARDS_DICT
    if REWARDS_DICT["ImageReward"] is None:
        REWARDS_DICT["ImageReward"] = rm_load("ImageReward-v1.0")

    with torch.no_grad():
        if image_tensors is not None:
            tensor_batch = image_tensors
        elif isinstance(images, torch.Tensor):
            tensor_batch = images
        elif isinstance(images, list) and images and isinstance(images[0], torch.Tensor):
            tensor_batch = torch.stack(images, dim=0)
        else:
            tensor_batch = None

        if tensor_batch is not None:
            out = REWARDS_DICT["ImageReward"].score_batched_tensor(
                prompts,
                tensor_batch,
                from_minus_one_to_one=from_minus_one_to_one,
                emulate_pil_uint8_roundtrip=emulate_pil_uint8_roundtrip,
                return_profile=return_profile,
            )
            return out

        image_reward_result = REWARDS_DICT["ImageReward"].score_batched(prompts, images)
        return image_reward_result

# Compute CLIP-Score
def do_clip_score(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["Clip-Score"] is None:
        REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=".", device="cuda")
    with torch.no_grad():
        clip_result = [
            REWARDS_DICT["Clip-Score"].score(prompt, images[i])
            for i, prompt in enumerate(prompts)
        ]
    return clip_result


# Compute LLM-grading
def do_llm_grading(*, images, prompts, metric_to_chase="overall_score"):
    global REWARDS_DICT
    
    if REWARDS_DICT["LLMGrader"] is None:
        REWARDS_DICT["LLMGrader"]  = LLMGrader()
    llm_grading_result = [
        REWARDS_DICT["LLMGrader"].score(images=images[i], prompts=prompt, metric_to_chase=metric_to_chase)
        for i, prompt in enumerate(prompts)
    ]
    return llm_grading_result


'''
@File       :   CLIPScore.py
@Time       :   2023/02/12 13:14:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   CLIPScore.
* Based on CLIP code base
* https://github.com/openai/CLIP
'''


class CLIPScore(nn.Module):
    def __init__(self, download_root, device='cpu'):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(
            "ViT-L/14", device=self.device, jit=False, download_root=download_root
        )

        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(
                self.clip_model
            )  # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)

    def score(self, prompt, pil_image, return_feature=False):
        # if (type(image_path).__name__=='list'):
        #     _, rewards = self.inference_rank(prompt, image_path)
        #     return rewards

        # text encode
        text = clip.tokenize(prompt, truncate=True).to(self.device)
        txt_features = F.normalize(self.clip_model.encode_text(text))

        # image encode
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = F.normalize(self.clip_model.encode_image(image))

        # score
        rewards = torch.sum(
            torch.mul(txt_features, image_features), dim=1, keepdim=True
        )

        if return_feature:
            return rewards, {'image': image_features, 'txt': txt_features}

        return rewards.detach().cpu().numpy().item()
