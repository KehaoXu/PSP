from importlib import resources
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import math
from torch.utils.checkpoint import checkpoint
from diffusers_patch.utils import TemperatureScaler
import torchvision
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer


ASSETS_PATH = resources.files("assets")

def classify_aesthetic_scores_easy(y):
    # Applying thresholds to map scores to classes
    class_labels = torch.zeros_like(y, dtype=torch.long)  # Ensure it's integer type for class labels
    class_labels[y >= 5.7] = 1
    class_labels[y < 5.7] = 0
    if class_labels.dim() > 1:
        return class_labels.squeeze(1)
    return class_labels

class SinusoidalTimeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 768  # Original input dimension
        self.time_encoding_dim = 768  # Dimension of time encoding
        self.concatenated_dim = self.input_dim + self.time_encoding_dim  # Total dimension after concatenation

        self.layers = nn.Sequential(
            nn.Linear(self.concatenated_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def sinusoidal_encoding(self, timesteps):
        # Normalize timesteps to be in the range [0, 1]
        timesteps = timesteps.float() / 1000.0  # Assuming timesteps are provided as integers
    
        # Generate a series of frequencies
        frequencies = torch.exp(torch.arange(0, self.time_encoding_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / self.time_encoding_dim))
        frequencies = frequencies.to(timesteps.device)

        # Apply the frequencies to the timesteps
        arguments = timesteps[:, None] * frequencies[None, :]
        encoding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=1)
        return encoding

    def forward(self, embed, timesteps):
        # Generate sinusoidal embeddings for the timesteps
        timestep_embed = self.sinusoidal_encoding(timesteps)

        # Concatenate the timestep embedding with the input tensor
        combined_input = torch.cat([embed, timestep_embed], dim=1)

        # Pass the combined input through the layers
        return self.layers(combined_input)

class MLPDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )


    def forward(self, embed):
        return self.layers(embed)
    
    def forward_up_to_second_last(self, embed):
        # Process the input through all layers except the last one
        for layer in list(self.layers)[:-1]:
            embed = layer(embed)
        return embed

class AestheticScorerDiff(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLPDiff()
        state_dict = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"), weights_only=True)
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    def __call__(self, images):
        device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1), embed
    
    def generate_feats(self, images):
        device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return embed

class hpsScorer(torch.nn.Module):
    def __init__(self, inference_dtype=None, device=None):
        super().__init__()
        self.device = device
        model_name = "ViT-H-14"
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            model_name,
            'laion2B-s32B-b79K',
            precision=inference_dtype,
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )

        tokenizer = get_tokenizer(model_name)

        link = "https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt"
        import os
        import requests
        from tqdm import tqdm

        # Create the directory if it doesn't exist
        os.makedirs(os.path.expanduser('~/.cache/hpsv2'), exist_ok=True)
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/hpsv2/HPS_v2_compressed.pt"

        # Download the file if it doesn't exist
        if not os.path.exists(checkpoint_path):
            response = requests.get(link, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(checkpoint_path, 'wb') as file, tqdm(
                    desc="Downloading HPS_v2_compressed.pt",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    progress_bar.update(size)

        # force download of model via score
        hpsv2.score([], "")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        self.tokenizer = get_tokenizer(model_name)
        model = model.to(device, dtype=inference_dtype)
        self.model = model
        self.model.eval()

        self.target_size = 224
        self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                     std=[0.26862954, 0.26130258, 0.27577711])

    # def score_fn(im_pix, prompts):
    def __call__(self, x_var, prompts, processed=True):
        if not processed:
            im_pix = x_var
            im_pix = ((im_pix / 2) + 0.5).clamp(0, 1)
            x_var = torchvision.transforms.Resize(self.target_size, antialias=False)(im_pix)
            x_var = self.normalize(x_var).to(im_pix.dtype)
        caption = self.tokenizer(prompts)
        caption = caption.to(self.device)
        outputs = self.model(x_var, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        return scores, None
        # loss = 1.0 - scores
        # return loss, scores
    # return score_fn


class AestheticScorerDiff_Time(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        # self.mlp = torch.load('aes_model/reward_predictor_epoch_3.pth')
        # self.mlp = torch.load('aes_model/reward_predictor_epoch_5_iter_4000.pth')
        self.mlp = torch.load('aes_model/reward_predictor_epoch_9.pth')
        self.dtype = dtype
        self.eval()
    
    def set_valuefunction(self, pathtomodel):
        self.mlp = torch.load(pathtomodel)
        print('Value function loaded: ', pathtomodel)
        self.mlp.eval()

    def __call__(self, images, timesteps): # timesteps: torch.randint(low=0, high=50, size=(32,))
        device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed, timesteps).squeeze(1), embed
    
    def generate_feats(self, images):
        device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return embed

class MLPDiff_class(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, out_channels),
        )

    def forward(self, embed):
        return self.layers(embed)

class condition_AestheticScorerDiff(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.dtype = dtype
        
        state_dict = torch.load('aesthetic_models/MLP_3class_easy_v1_final_calibrated.pth')

        self.scaler = TemperatureScaler()
        self.scaler.load_state_dict(state_dict['scaler'])
        
        self.mlp = MLPDiff_class(out_channels=3)
        self.mlp.load_state_dict(state_dict['model_state_dict'])
        
        self.eval()

    def __call__(self, images):
        device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        
        logits = self.mlp(embed)
        calibrated_logits = self.scaler(logits)
        probabilities = F.softmax(calibrated_logits, dim=1)
        
        return probabilities, embed


if __name__ == "__main__":
    model = SinusoidalTimeMLP()
    embed = torch.randn(32, 768)
    timesteps = torch.randint(low=0, high=50, size=(32,))
    
    print(model.sinusoidal_encoding(timesteps).shape)
    print(model(embed, timesteps).shape)