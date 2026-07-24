from sd_pipeline import DPS_continuous_SDPipeline,  Decoding_SDPipeline, SMC_SDPipeline
from diffusers import DDIMScheduler
import torch
import numpy as np
import random
from PIL import Image
import PIL
from typing import Callable, List, Optional, Union, Dict, Any
from dataset import AVACompressibilityDataset, AVACLIPDataset, AVAHpsDataset
from vae import encode
import os
from aesthetic_scorer import AestheticScorerDiff_Time, MLPDiff
import wandb
import argparse
from tqdm import tqdm
import datetime
from compressibility_scorer import CompressibilityScorerDiff, jpeg_compressibility, CompressibilityScorer_modified
from aesthetic_scorer import AestheticScorerDiff
from transformers import CLIPProcessor, CLIPModel
from brisque import BRISQUE
from aesthetic_scorer import AestheticScorerDiff, hpsScorer
from utils import compute_metrics

def parse():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward", type=str, default='aesthetic')
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--n_size", type=int, default=25)
    parser.add_argument("--num_group", type=int, default=3)
    parser.add_argument("--val_bs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duplicate_size",type=int, default=20)
    parser.add_argument("--variant", type=str, default="PM")
    parser.add_argument("--valuefunction", type=str, default="")
    parser.add_argument("--alpha", type=float, default= 20.0)
    parser.add_argument("--wandb_mode", type=str, default="online")
    args = parser.parse_args()
    return args


######### preparation ##########

args = parse()
device= args.device
save_file = True

assert args.num_images % args.num_group == 0
args.num_images *= args.n_size

## Image Seeds
if args.seed > 0:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    shape = (args.num_images//args.bs, args.bs , 4, 64, 64)
    init_latents = torch.randn(shape, device=device)
else:
    init_latents = None

run_name = f"{args.variant}_reward_{args.reward}_{args.valuefunction.split('/')[-1] if args.valuefunction != '' else ''}"
unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
run_name = run_name + '_' + unique_id + '_SMC'


if args.out_dir == "":
    args.out_dir = 'logs/' + run_name
try:
    os.makedirs(args.out_dir)
except:
    pass


wandb.init(project=f"SMC-{args.reward}", name=run_name,config=args, mode=args.wandb_mode)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
initial_memory = torch.cuda.memory_allocated()

sd_model = SMC_SDPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", local_files_only=True)
sd_model.to(device)

# switch to DDIM scheduler
sd_model.scheduler = DDIMScheduler.from_config(sd_model.scheduler.config)
sd_model.scheduler.set_timesteps(50, device=device)

sd_model.vae.requires_grad_(False)
sd_model.text_encoder.requires_grad_(False)
sd_model.unet.requires_grad_(False)

sd_model.vae.eval()
sd_model.text_encoder.eval()
sd_model.unet.eval()

assert args.variant in ['PM', 'MC']

if args.reward == 'compressibility':
    if args.variant == 'PM':
        scorer = CompressibilityScorer_modified(dtype=torch.float32)#.to(device)
    elif args.variant == 'MC':
        scorer = CompressibilityScorerDiff(dtype=torch.float32).to(device)
elif args.reward == 'aesthetic':
    if args.variant == 'PM':
        scorer = AestheticScorerDiff(dtype=torch.float32).to(device)
    elif args.variant == 'MC':
        scorer = AestheticScorerDiff_Time(dtype=torch.float32).to(device)
        if args.valuefunction != "":
            scorer.set_valuefunction(args.valuefunction)
            scorer = scorer.to(device)
elif args.reward == 'hps':
    if args.variant == 'PM':
        scorer = hpsScorer(inference_dtype=torch.float32, device=device).to(device)
    else:
        raise ValueError("Invalid variant")
else:
    raise ValueError("Invalid reward")

scorer.requires_grad_(False)
scorer.eval()

sd_model.setup_scorer(scorer)
sd_model.set_variant(args.variant)
sd_model.set_reward(args.reward)
sd_model.set_parameters(args.bs, args.duplicate_size, args.alpha)

### introducing evaluation prompts
import prompts as prompts_file
if args.reward == 'hps':
    eval_prompt_fn = getattr(prompts_file, 'eval_hps_v2')
else:
    eval_prompt_fn = getattr(prompts_file, 'eval_aesthetic_SMC_animals')
    batchwise_prompts = ['dog', 'cat', 'panda', 'monkey', 'rabbit', 'butterfly', 'horse']


image = []
eval_prompt_list = []
KL_list = []

for i in tqdm(range(args.num_images // args.bs), desc="Generating Images"):
    wandb.log(
        {"inner_iter": i}
    )
    if init_latents is None:
        init_i = None
    else:
        init_i = init_latents[i]
    eval_prompts, _ = zip(
        *[eval_prompt_fn() for _ in range(args.bs)]
    )
    eval_prompts = list(eval_prompts)
    eval_prompts = [eval_prompts[i % 7] for _ in range(args.bs)]
    eval_prompt_list.extend(eval_prompts)
    
    image_, kl_loss = sd_model(eval_prompts, num_images_per_prompt=1, eta=1.0, latents=init_i) # List of PIL.Image objects
    image.extend(image_)
    KL_list.append(kl_loss)

# KL_entropy = torch.mean(torch.stack(KL_list))

end_event.record()
torch.cuda.synchronize() # Wait for the events to complete
gpu_time = start_event.elapsed_time(end_event)/1000 # Time in seconds
max_memory = torch.cuda.max_memory_allocated()
max_memory_used = (max_memory - initial_memory) / (1024 ** 2)

wandb.log({
        "GPUTimeInS": gpu_time,
        "MaxMemoryInMb": max_memory_used,
    })

###### evaluation and metric #####
# CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
if args.reward == 'compressibility':
    gt_dataset= AVACompressibilityDataset(image)
elif args.reward == 'aesthetic':
    from importlib import resources
    ASSETS_PATH = resources.files("assets")
    eval_model = MLPDiff().to(device)
    eval_model.requires_grad_(False)
    eval_model.eval()
    s = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"), map_location=device, weights_only=True)
    eval_model.load_state_dict(s)
    gt_dataset= AVACLIPDataset(image)    
elif args.reward == 'hps':
    gt_dataset= AVAHpsDataset(image)

gt_dataloader = torch.utils.data.DataLoader(gt_dataset, batch_size=args.val_bs, shuffle=False)

with torch.no_grad():
    eval_rewards = []
    all_image_embeds = []
    all_quality_score = []
    for image_idx, inputs in enumerate(gt_dataloader):
        inputs = inputs.to(device)

        if args.reward == 'compressibility':
            jpeg_compressibility_scores = jpeg_compressibility(inputs)
            scores = torch.tensor(jpeg_compressibility_scores, dtype=inputs.dtype, device=inputs.device)
        
        elif args.reward == 'aesthetic':
            scores = eval_model(inputs)
            scores = scores.squeeze(1)
        elif args.reward == 'hps':
            scores, _ = scorer(inputs, [eval_prompt_list[image_idx]], processed=False)

        # record embedding
        raw_image = image[image_idx]
        inputs_clip = processor(images=raw_image, return_tensors="pt")
        inputs_clip = {key: value.to(device) for key, value in inputs_clip.items()}
        image_embed = clip_model.get_image_features(**inputs_clip)  # bs * 512
        all_image_embeds.append(image_embed.cpu())

        # quality score
        obj = BRISQUE(url=False)
        quality_score = obj.score(img=np.asarray(image[image_idx]))
        all_quality_score.append(max(quality_score, 0))

        # reward
        eval_rewards.extend(scores.tolist())

eval_rewards = torch.tensor(eval_rewards)
if save_file:
    images = []
    log_dir = os.path.join(args.out_dir, "eval_vis")
    os.makedirs(log_dir, exist_ok=True)
    np.save(f"{args.out_dir}/scores.npy", eval_rewards)

    # Function to save array to a text file with commas
    def save_array_to_text_file(array, file_path):
        with open(file_path, 'w') as file:
            array_str = ','.join(map(str, array.tolist()))
            file.write(array_str + ',')

    # Save the arrays to text files
    save_array_to_text_file(eval_rewards, f"{args.out_dir}/eval_rewards.txt")
    print("Arrays have been saved to text files.")
    
    for idx, im in enumerate(image):
        prompt = eval_prompt_list[idx]
        reward = eval_rewards[idx]
        
        im.save(f"{log_dir}/{idx:03d}_{prompt}_score={reward:2f}.png")
        
        pil = im.resize((256, 256))

        images.append(wandb.Image(pil, caption=f"{prompt:.25} | score:{reward:.2f}"))

    wandb.log(
        {"images": images}
    )
