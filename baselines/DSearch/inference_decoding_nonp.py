from sd_pipeline import DPS_continuous_SDPipeline, Decoding_nonbatch_SDPipeline
from diffusers import DDIMScheduler
import torch
import torch.nn.functional as F
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
from aesthetic_scorer import AestheticScorerDiff, hpsScorer
from transformers import CLIPProcessor, CLIPModel
from brisque import BRISQUE
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from sklearn.metrics.pairwise import cosine_similarity


def parse():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward", type=str, default='aesthetic')
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--oversamplerate", type=int, default=1)
    parser.add_argument("--w", type=float, default=1)
    parser.add_argument("--search_schudule", type=str, default="all")
    parser.add_argument("--drop_schudule", type=str, default=None)
    parser.add_argument("--replacerate", type=float, default=0)
    parser.add_argument("--val_bs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duplicate_size",type=int, default=20)  
    parser.add_argument("--variant", type=str, default="PM")
    parser.add_argument("--valuefunction", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online")
    args = parser.parse_args()
    return args


######### preparation ##########

args = parse()
device= args.device
save_file = True

# DDES type
DDES_type = None
assert args.oversamplerate == 1 or args.replacerate == 0.0
if args.oversamplerate > 1:
    DDES_type = 'DDES_E'
elif args.replacerate > 0:
    DDES_type = 'DDES_R'
else:
    DDES_type = 'SVDD'
# assert args.num_images // args.bs == 3
num_group = 3  #args.num_images // args.bs

## Image Seeds
if args.seed > 0:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    shape = (args.num_images//args.bs, (args.bs*args.oversamplerate) , 4, 64, 64)
    init_latents = torch.randn(shape, device=device)
else:
    init_latents = None

if args.replacerate <= 0 and args.oversamplerate <= 1:
    search_appx = ""
else:
    search_appx = f"R{args.replacerate}" if args.replacerate> 0 else f"E{args.oversamplerate}"
    search_appx = search_appx + f"_C{args.w*args.oversamplerate}_{args.search_schudule}_{args.drop_schudule}"
run_name = f"{args.variant}_M={args.duplicate_size}_reward_{args.reward}_{args.valuefunction.split('/')[-1] if args.valuefunction != '' else ''}" + search_appx
unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
run_name = run_name + '_' + unique_id


if args.out_dir == "":
    args.out_dir = 'logs/' + run_name
try:
    os.makedirs(args.out_dir)
except:
    pass


wandb.init(project=f"SVDD-{args.reward}", name=run_name, config=args, mode=args.wandb_mode)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
initial_memory = torch.cuda.memory_allocated()

sd_model = Decoding_nonbatch_SDPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", local_files_only=False)
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
sd_model.set_parameters(args.bs, args.duplicate_size, args.w, args.search_schudule, args.drop_schudule, args.oversamplerate, args.replacerate)

### introducing evaluation prompts
import prompts as prompts_file
if args.reward == 'hps':
    eval_prompt_fn = getattr(prompts_file, 'eval_hps_v2')
else:
    eval_prompt_fn = getattr(prompts_file, 'eval_aesthetic_animals')
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
        *[eval_prompt_fn() for _ in range(args.bs*args.oversamplerate)]
    )
    eval_prompts = list(eval_prompts)
    if search_appx != "" and search_appx.startswith("R"):
        print("search_appx", search_appx)
        eval_prompts = [eval_prompts[i%7] for _ in range(args.bs * args.oversamplerate)]
    
    image_, kl_loss, cur_prompt = sd_model(eval_prompts, num_images_per_prompt=1, eta=1.0, latents=init_i) # List of PIL.Image objects
    if search_appx != "" and search_appx.startswith("E"):
        print("search_appx", search_appx)
        eval_prompt_list.extend(cur_prompt)
    else:
        eval_prompt_list.extend(eval_prompts)
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


def compute_metrics(r_batch):
    return_dict = {}
    for key in r_batch:
        value = r_batch[key]
        if key == 'diversity':
            value = torch.concat(value, dim=0)  # bs * dim
            # value = F.normalize(value, p=2, dim=1)
            cosine_matrix = cosine_similarity(value)
            # cosine_matrix = torch.mm(value, value.T)

            bs = cosine_matrix.shape[0]
            cosine_matrix_no_diag = cosine_matrix - np.eye(bs)
            cosine_similarity_matrix = cosine_matrix_no_diag.sum() / (bs * (bs - 1) + 1e-8)
            mean_value = 1 - cosine_similarity_matrix.item()
        else:
            mean_value = np.mean(value).item()
        return_dict[f'{key}'] = mean_value
    return return_dict


# CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# clip_model, processor = clip.load('ViT-B/32', device)
# image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
# vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", device_map=device)

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

        # image_input = processor(image).unsqueeze(0).to(device)
        # image_embed = clip_model.encode_image(image_input)
        # all_image_embeds.append(image_embed)

        # inputs_clip = image_processor.preprocess(inputs, return_tensors='pt')['pixel_values'][0]
        # inputs_clip = inputs_clip.to(device)
        # image_embed = vision_tower(inputs_clip.unsqueeze(0), output_hidden_states=True)
        # image_embed = image_embed.hidden_states[-2]
        # image_embed = image_embed.reshape(1, -1)
        # all_image_embeds.append(image_embed)

        # image_embed = inputs.reshape(1, -1)
        # all_image_embeds.append(image_embed)

        # quality score
        obj = BRISQUE(url=False)
        quality_score = obj.score(img=np.asarray(image[image_idx]))
        all_quality_score.append(quality_score)

        # reward
        eval_rewards.extend(scores.tolist())

    assert len(eval_rewards) == len(all_image_embeds) == len(all_quality_score)
    # shuffle
    combined = list(zip(eval_rewards, all_image_embeds, all_quality_score))
    random.shuffle(combined)
    eval_rewards1, all_image_embeds, all_quality_score = zip(*combined)
    eval_rewards1 = list(eval_rewards1)
    all_image_embeds = list(all_image_embeds)
    all_quality_score = list(all_quality_score)

    # split list to 3 groups
    n = len(eval_rewards1) // num_group
    result_batches = [
        {
            "rewards": eval_rewards1[i * n:(i + 1) * n],
            "diversity": all_image_embeds[i * n:(i + 1) * n],
            "quality": all_quality_score[i * n:(i + 1) * n]
        }
        for i in range(num_group)
    ]

    # each group metrics
    result_batches_metrics = [compute_metrics(batch) for batch in result_batches]

    log_metrics = {}
    mean_metrics = {f"eval_{args.reward}_{key}_mean": np.mean([m[key] for m in result_batches_metrics]) for key in result_batches_metrics[0]}
    std_metrics = {f"eval_{args.reward}_{key}_std": np.std([m[key] for m in result_batches_metrics]) for key in result_batches_metrics[0]}
    log_metrics.update(mean_metrics)
    log_metrics.update(std_metrics)

    # # calculate diversity
    # all_image_embeds = torch.concat(all_image_embeds, dim=0)  # bs * dim
    # all_image_embeds = F.normalize(all_image_embeds, p=2, dim=1)
    # cosine_similarity_matrix = torch.mm(all_image_embeds, all_image_embeds.T)
    #
    # bs = cosine_similarity_matrix.size(0)
    # cosine_similarity_matrix_no_diag = cosine_similarity_matrix - torch.eye(bs, device=cosine_similarity_matrix.device)
    # mean_cosine_similarity = cosine_similarity_matrix_no_diag.sum() / (bs * (bs - 1))
    # diversity = 1 - mean_cosine_similarity.item()
    #
    # # quality
    # quality = sum(all_quality_score) / len(all_quality_score)

    eval_rewards1 = torch.tensor(eval_rewards1)
    print(f"eval_{args.reward}_rewards_mean: {torch.mean(eval_rewards1)}")

    # wandb.log({
    #     f"eval_{args.reward}_rewards_mean": torch.mean(eval_rewards),
    #     f"eval_{args.reward}_diversity": diversity,
    #     f"eval_{args.reward}_quality": quality,
    # })
    wandb.log(log_metrics)

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
    save_array_to_text_file(torch.tensor(eval_rewards), f"{args.out_dir}/eval_rewards.txt")
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
