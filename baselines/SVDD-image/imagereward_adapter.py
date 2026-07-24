from pathlib import Path
import sys

import torch
from PIL import Image


_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def _resolve_fk_text_to_image_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "Fk-Diffusion-Steering" / "text_to_image"


class ImageRewardScorerAdapter(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        fk_path = _resolve_fk_text_to_image_path()
        if str(fk_path) not in sys.path:
            sys.path.append(str(fk_path))
        from fkd_diffusers.image_reward_utils import rm_load

        self.model = rm_load("ImageReward-v1.0", device=device)
        self.prompts = []
        self.device = device

    def set_prompts(self, prompts):
        self.prompts = list(prompts)

    def _match_prompts(self, batch_size):
        if not self.prompts:
            raise ValueError("ImageReward prompts are not set for current batch")
        if len(self.prompts) == batch_size:
            return self.prompts
        repeats = (batch_size + len(self.prompts) - 1) // len(self.prompts)
        return (self.prompts * repeats)[:batch_size]

    def _tensor_to_pil_list(self, image_tensor):
        mean = _CLIP_MEAN.to(image_tensor.device, image_tensor.dtype)
        std = _CLIP_STD.to(image_tensor.device, image_tensor.dtype)
        image_tensor = (image_tensor * std + mean).clamp(0.0, 1.0)
        image_tensor = image_tensor.detach().cpu()
        pil_images = []
        for sample in image_tensor:
            arr = (sample.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
            pil_images.append(Image.fromarray(arr))
        return pil_images

    def forward(self, image_tensor, timesteps=None):
        prompts = self._match_prompts(image_tensor.shape[0])
        pil_images = self._tensor_to_pil_list(image_tensor)
        scores = self.model.score_batched(prompts, pil_images)
        scores = torch.tensor(scores, device=image_tensor.device, dtype=image_tensor.dtype)
        return scores, None

    def score_pil(self, prompts, pil_images):
        return self.model.score_batched(list(prompts), list(pil_images))
