"""Shared code for the SD1.5 x GenEval trajectory cache (see README.md).

Step convention: step k = 1..T is the k-th UNet call. x0[k] is that call's pred_original_sample and
xt[k] is the latent after its DDIM update (the input of step k+1), so xt[T] is the final latent.
"""
import json
import os
import sys
from pathlib import Path

# read when the first cuBLAS handle is created, so it must be set before any CUDA work
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PSP_T2I = REPO_ROOT / "Fk-Diffusion-Steering" / "text_to_image"
GENEVAL_METADATA = PSP_T2I / "prompt_files" / "geneval_metadata.jsonl"
# fkd_class.py does `from smc_utils import ...`, so fkd_diffusers/ itself must be importable too
sys.path[:0] = [str(PSP_T2I), str(PSP_T2I / "fkd_diffusers")]

# runwayml/stable-diffusion-v1-5 was taken down from the Hub; this official mirror has the same weights
SD15_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
GUIDANCE_SCALE = 7.5  # FKDStableDiffusion.__call__ default, which PSP's scripts leave unchanged
LATENT_SHAPE = (4, 64, 64)
NUM_PROMPTS = 553
HPS_CHECKPOINT = ("xswu/HPSv2", "HPS_v2.1_compressed.pt")
BLIP_ITM_REPO = "Salesforce/blip-itm-large-coco"
SCORE_COLS = ["ir", "hps", "blip_itm", "blip_itc"]
EMB_DIMS = {"emb_hps": 1024, "emb_blip": 256}
SCORER_VERSIONS = {
    "ir": "ImageReward-v1.0 via PSP image_reward_utils.rm_load / score_batched, fp32",
    "hps": f"{HPS_CHECKPOINT[1]} via hpsv2 1.2.0 (ViT-H-14), autocast fp16, as hpsv2.score",
    "blip": f"{BLIP_ITM_REPO}, fp16; ITM = softmax(itm_head)[:, 1], ITC = cosine, emb = ITC image feature",
}


def set_determinism():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------- prompts, folds, layout

def load_geneval():
    """GenEval metadata rows; prompt_id = line number, as in PSP's collect script."""
    lines = [l for l in GENEVAL_METADATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [json.loads(l) for l in lines]
    assert len(rows) == NUM_PROMPTS, len(rows)
    return rows


def make_folds(rows, n_folds=5, seed=0):
    """Prompt-level CV folds stratified by GenEval tag: each tag's prompts are shuffled and dealt
    round-robin, the deal continuing across tags so fold sizes differ by at most one."""
    rng = np.random.default_rng(seed)
    fold_of, start = {}, 0
    for tag in sorted({r["tag"] for r in rows}):
        ids = [i for i, r in enumerate(rows) if r["tag"] == tag]
        for j, pid in enumerate(rng.permutation(ids).tolist()):
            fold_of[pid] = (start + j) % n_folds
        start += len(ids)
    folds = [sorted(p for p, f in fold_of.items() if f == k) for k in range(n_folds)]
    return {"n_folds": n_folds, "seed": seed, "stratify": "tag", "folds": folds}


def shard_prompt_ids(shard, num_shards, n=NUM_PROMPTS):
    """Contiguous blocks; 4 shards give 0-138, 139-276, 277-414, 415-552."""
    return [int(i) for i in np.array_split(np.arange(n), num_shards)[shard]]


def prompt_dir(cache, pid):
    return Path(cache) / "prompts" / f"{pid:04d}"


def png_name(seed):
    return f"seed_{seed:02d}.png"


def write_text_once(path, text):
    """Atomically write a file that every shard writes identically; an existing file must match."""
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"{path} exists with different content")
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- generation

def load_sd15(device):
    """PSP's fks_utils.get_model("stable-diffusion-v1-5") with the mirror repo: fp16 pipeline, DDIM."""
    from diffusers import DDIMScheduler
    from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion

    pipe = FKDStableDiffusion.from_pretrained(SD15_REPO, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def initial_noise(seeds, kind, device):
    """x_T for each seed, fp16 [N, 4, 64, 64].

    cpu: the cache's convention. One CPU torch.Generator per seed, drawn in fp32 and cast, so a seed's
    noise depends on nothing but the seed (not the batch, the seed count or the GPU).
    psp_cuda: what PSP's collect script gets from diffusers' randn_tensor with one CUDA generator per
    seed, drawn in fp16; only used to reproduce the midterm run.
    """
    if kind == "cpu":
        noise = [torch.randn(LATENT_SHAPE, generator=torch.Generator("cpu").manual_seed(s)) for s in seeds]
        return torch.stack(noise).half().to(device)
    if kind == "psp_cuda":
        noise = [
            torch.randn((1, *LATENT_SHAPE), generator=torch.Generator(device).manual_seed(s),
                        device=device, dtype=torch.float16)
            for s in seeds
        ]
        return torch.cat(noise)
    raise ValueError(kind)


@torch.inference_mode()
def sample(pipe, prompt, x_T, num_steps, xt_steps=(), guidance_scale=GUIDANCE_SCALE, eta=0.0):
    """DDIM from x_T with all seeds in one UNet batch; the same computation as FKDStableDiffusion.__call__
    (no negative prompt), which collect_cache.py --verify-pipeline checks.

    Returns x0 [N, T, 4, 64, 64], {k: xt[k] [N, 4, 64, 64] for k in xt_steps} and the T timesteps.
    """
    assert guidance_scale > 1
    cond, uncond = pipe.encode_prompt([prompt] * len(x_T), x_T.device, 1, True)
    embeds = torch.cat([uncond, cond])
    sched = pipe.scheduler
    sched.set_timesteps(num_steps, device=x_T.device)
    latents = x_T * sched.init_noise_sigma
    x0, xt = [], {}
    for k, t in enumerate(sched.timesteps, start=1):
        model_in = sched.scale_model_input(torch.cat([latents] * 2), t)
        eps_uncond, eps_cond = pipe.unet(model_in, t, encoder_hidden_states=embeds, return_dict=False)[0].chunk(2)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        out = sched.step(eps, t, latents, eta=eta, return_dict=True)
        latents = out.prev_sample
        x0.append(out.pred_original_sample)
        if k in xt_steps:
            xt[k] = latents
    return torch.stack(x0, dim=1), xt, [int(t) for t in sched.timesteps]


@torch.inference_mode()
def decode_to_pil(pipe, latents, batch_size):
    """VAE decode -> clamp to [-1, 1] -> uint8 -> PIL (the PSP pipeline's output path), batch_size at a time."""
    from fkd_diffusers.fkd_pipeline_sd import latent_to_decode

    images = []
    for i in range(0, len(latents), batch_size):
        decoded = latent_to_decode(model=pipe, output_type="pil", latents=latents[i:i + batch_size])
        images += pipe.image_processor.postprocess(decoded, output_type="pil")
    return images


# ---------------------------------------------------------------- scorers (all take PIL images of one prompt)

class ImageRewardScorer:
    """PSP's ImageReward wrapper as PSP calls it (score_batched on PIL images, fp32): the IR of the PSP runs."""

    def __init__(self, device):
        from fkd_diffusers.image_reward_utils import rm_load

        self.model = rm_load("ImageReward-v1.0", device=device)

    @torch.inference_mode()
    def __call__(self, prompt, images):
        return np.asarray(self.model.score_batched([prompt] * len(images), images), dtype=np.float64)


class HPSScorer:
    """HPSv2.1 computed as hpsv2.score does (same model, transform and autocast), but loaded once and
    run on a batch: hpsv2.score reloads the 2 GB checkpoint on every call."""

    def __init__(self, device):
        import huggingface_hub
        from fkd_diffusers.rewards import _install_turtle_shim_for_hpsv2

        _install_turtle_shim_for_hpsv2()
        from hpsv2 import img_score
        from hpsv2.src.open_clip import get_tokenizer

        img_score.initialize_model()
        self.model = img_score.model_dict["model"]
        ckpt = huggingface_hub.hf_hub_download(*HPS_CHECKPOINT)
        self.model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
        self.model.to(device).eval()
        self.preprocess = img_score.model_dict["preprocess_val"]
        self.tokenizer = get_tokenizer("ViT-H-14")
        self.device = device

    @torch.inference_mode()
    def __call__(self, prompt, images):
        """-> scores [B], L2-normalized image embeddings [B, 1024] fp16."""
        image = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        text = self.tokenizer([prompt]).to(self.device)
        with torch.autocast(device_type="cuda"):
            out = self.model(image, text)
            scores = (out["image_features"] @ out["text_features"].T)[:, 0]
        return scores.float().cpu().numpy().astype(np.float64), out["image_features"].half().cpu().numpy()


class BlipITMScorer:
    """Salesforce/blip-itm-large-coco in fp16. Same computation as BlipForImageTextRetrieval.forward, written
    out so the vision tower runs once for both heads, and because transformers 4.38.2 builds the ITM
    cross-attention mask on the CPU, which fails on a GPU."""

    def __init__(self, device):
        from transformers import BlipForImageTextRetrieval, BlipProcessor

        self.processor = BlipProcessor.from_pretrained(BLIP_ITM_REPO)
        self.model = BlipForImageTextRetrieval.from_pretrained(BLIP_ITM_REPO, torch_dtype=torch.float16)
        self.model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, prompt, images):
        """-> ITM match probability [B], ITC cosine [B], normalized ITC image embeddings [B, 256] fp16."""
        m = self.model
        inputs = self.processor(images=images, text=[prompt] * len(images), return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        text_kwargs = dict(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        image_embeds = m.vision_model(pixel_values=inputs.pixel_values.to(m.dtype))[0]
        image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=self.device)
        itm = m.text_encoder(**text_kwargs, encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts)[0]
        itm_prob = m.itm_head(itm[:, 0, :]).float().softmax(dim=-1)[:, 1]
        text_embeds = m.text_encoder(**text_kwargs)[0]
        image_feat = F.normalize(m.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_feat = F.normalize(m.text_proj(text_embeds[:, 0, :]), dim=-1)
        itc = (image_feat @ text_feat.t()).diagonal()
        return (itm_prob.cpu().numpy().astype(np.float64), itc.float().cpu().numpy().astype(np.float64),
                image_feat.cpu().numpy())


class Scorers:
    def __init__(self, device):
        self.ir, self.hps, self.blip = ImageRewardScorer(device), HPSScorer(device), BlipITMScorer(device)

    def __call__(self, prompt, images, batch_size):
        """-> {ir, hps, blip_itm, blip_itc: [N] float64, emb_hps: [N, 1024], emb_blip: [N, 256] fp16}."""
        parts = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            hps, emb_hps = self.hps(prompt, batch)
            itm, itc, emb_blip = self.blip(prompt, batch)
            parts.append(dict(ir=self.ir(prompt, batch), hps=hps, blip_itm=itm, blip_itc=itc,
                              emb_hps=emb_hps, emb_blip=emb_blip))
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
