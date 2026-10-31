#!/usr/bin/env python3
"""Fill the SD1.5 x GenEval trajectory cache for one block of prompts (one process per GPU).

Per prompt: DDIM-sample all seeds in one UNet batch from the shared x_T, decode x0 at every
--score-every-th step and score it with IR / HPS / BLIP-ITM, decode and score the final images, and
write prompts/<pid>/ (layout in README.md). Each prompt is written under a temporary name and renamed
once complete, with DONE inside, so a rerun skips finished prompts and never sees half a prompt.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from cache_lib import (
    GUIDANCE_SCALE, NUM_PROMPTS, PSP_T2I, REPO_ROOT, SCORER_VERSIONS, SD15_REPO, Scorers, decode_to_pil,
    initial_noise, load_geneval, load_sd15, make_folds, png_name, prompt_dir, sample, set_determinism,
    shard_prompt_ids, write_text_once,
)
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
DEVICE = "cuda"
PSP_FILES_USED = ["fkd_diffusers/fkd_pipeline_sd.py", "fkd_diffusers/image_reward_utils.py", "fkd_diffusers/rewards.py"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="cache root, e.g. /workspace/cache_sd15_geneval_t32")
    which = p.add_mutually_exclusive_group()
    which.add_argument("--shard", type=int, help="block --shard of --num-shards contiguous blocks of prompt ids")
    which.add_argument("--prompt-ids", type=int, nargs="+")
    which.add_argument("--prompt-start-id", type=int)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--num-prompts", type=int, help="with --prompt-start-id")
    p.add_argument("--num-seeds", type=int, default=32)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--num-steps", type=int, default=32)
    p.add_argument("--score-every", type=int, default=2)
    p.add_argument("--x0-steps", choices=["all", "scored"], default="all", help="steps whose x0 latent is saved")
    p.add_argument("--xt-steps", choices=["all", "scored"], default="scored", help="steps whose x_t latent is saved")
    p.add_argument("--decode-batch", type=int, default=8)
    p.add_argument("--score-batch", type=int, default=32)
    p.add_argument("--verify-pipeline", action="store_true",
                   help="before collecting, check the sampler against PSP's pipeline on the first prompt")
    return p.parse_args()


def selected_prompt_ids(args):
    if args.shard is not None:
        return shard_prompt_ids(args.shard, args.num_shards)
    if args.prompt_ids:
        return args.prompt_ids
    start = args.prompt_start_id or 0
    end = NUM_PROMPTS if args.num_prompts is None else min(NUM_PROMPTS, start + args.num_prompts)
    return list(range(start, end))


def build_config(args):
    """Everything that determines a prompt's outputs; a resumed run must match it exactly."""
    assert args.num_steps % args.score_every == 0, "the last step must be a scored step"
    steps = list(range(1, args.num_steps + 1))
    score_steps = steps[args.score_every - 1::args.score_every]
    return {
        "model": SD15_REPO, "dtype": "fp16", "sampler": "DDIM", "num_steps": args.num_steps, "eta": 0.0,
        "guidance_scale": GUIDANCE_SCALE, "negative_prompt": None,
        "seeds": list(range(args.base_seed, args.base_seed + args.num_seeds)),
        "noise": "x_T[s] = randn(4, 64, 64) from torch.Generator('cpu').manual_seed(s), fp32 -> fp16; same for all prompts",
        "score_steps": score_steps,
        "x0_steps": steps if args.x0_steps == "all" else score_steps,
        "xt_steps": steps if args.xt_steps == "all" else score_steps,
        "unet_batch": args.num_seeds, "decode_batch": args.decode_batch, "score_batch": args.score_batch,
        "scorers": SCORER_VERSIONS,
    }


def is_done(out, pid, config):
    done = prompt_dir(out, pid) / "DONE"
    if not done.exists():
        return False
    if json.loads(done.read_text())["config"] != json.loads(json.dumps(config)):
        raise SystemExit(f"{done.parent} was collected with a different config; use another --out")
    return True


def run_meta(run_id, args, config, prompt_ids):
    def cmd(*argv):
        try:
            return subprocess.run(argv, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    from huggingface_hub import try_to_load_from_cache

    sd15_index = try_to_load_from_cache(SD15_REPO, "model_index.json")
    code = {f"exps/cache_sd15_geneval_t32/{p.name}": p for p in sorted(HERE.glob("*.py"))}
    code.update({f"Fk-Diffusion-Steering/text_to_image/{f}": PSP_T2I / f for f in PSP_FILES_USED})
    return {
        "run_id": run_id,
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "argv": sys.argv,
        "args": vars(args),
        "config": config,
        "prompt_ids": prompt_ids,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
            "driver": cmd("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "numerics": {
            "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "sd15_snapshot": Path(sd15_index).parent.name if isinstance(sd15_index, str) else None,
        "git": {"commit": cmd("git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"),
                "status": cmd("git", "-C", str(REPO_ROOT), "status", "--porcelain")},
        "code_md5": {name: hashlib.md5(path.read_bytes()).hexdigest() for name, path in code.items()},
        "pip_freeze": (cmd(sys.executable, "-m", "pip", "freeze") or "").splitlines(),
    }


def compare_with_pipeline(pipe, prompt, x_T, num_steps):
    """Max |diff| between this sampler's final latent and PSP's FKDStableDiffusion.__call__ from the same x_T."""
    _, xt, _ = sample(pipe, prompt, x_T, num_steps, xt_steps={num_steps})
    with torch.inference_mode():
        ref = pipe([prompt] * len(x_T), num_inference_steps=num_steps, guidance_scale=GUIDANCE_SCALE, eta=0.0,
                   latents=x_T.clone(), output_type="latent").images
    return (xt[num_steps].float() - ref.float()).abs().max().item()


def collect_prompt(pipe, scorers, out, pid, row, config, run_id):
    t0 = time.time()
    prompt, tag, seeds, n_steps = row["prompt"].strip(), row["tag"], config["seeds"], config["num_steps"]
    x_T = initial_noise(seeds, kind="cpu", device=DEVICE)
    x0, xt, timesteps = sample(pipe, prompt, x_T, n_steps, xt_steps=set(config["xt_steps"]))

    parts, emb_hps, emb_blip = [], [], []
    for k in config["score_steps"]:
        s = scorers(prompt, decode_to_pil(pipe, x0[:, k - 1], config["decode_batch"]), config["score_batch"])
        emb_hps.append(s.pop("emb_hps"))
        emb_blip.append(s.pop("emb_blip"))
        parts.append(pd.DataFrame({"seed": seeds, "step": k, "timestep": timesteps[k - 1], **s}))
    scores = pd.concat(parts).sort_values(["seed", "step"], kind="stable").reset_index(drop=True)
    scores.insert(0, "category", tag)
    scores.insert(0, "prompt_id", pid)

    final_images = decode_to_pil(pipe, xt[n_steps], config["decode_batch"])
    f = scorers(prompt, final_images, config["score_batch"])
    final = pd.DataFrame({"prompt_id": pid, "category": tag, "seed": seeds,
                          **{k: v for k, v in f.items() if not k.startswith("emb_")}})

    tmp = out / "prompts" / f".tmp-{pid:04d}-{run_id}"
    (tmp / "final").mkdir(parents=True)
    np.save(tmp / "xT.npy", x_T.cpu().numpy())
    np.save(tmp / "x0_latents.npy", x0[:, [k - 1 for k in config["x0_steps"]]].cpu().numpy())
    np.save(tmp / "xt_latents.npy", torch.stack([xt[k] for k in config["xt_steps"]], dim=1).cpu().numpy())
    np.save(tmp / "emb_hps.npy", np.stack(emb_hps, axis=1))
    np.save(tmp / "emb_blip.npy", np.stack(emb_blip, axis=1))
    scores.to_parquet(tmp / "scores.parquet", index=False)
    final.to_parquet(tmp / "final.parquet", index=False)
    for seed, image in zip(seeds, final_images):
        image.save(tmp / "final" / png_name(seed))
    (tmp / "DONE").write_text(json.dumps({
        "prompt_id": pid, "prompt": prompt, "category": tag, "timesteps": timesteps, "config": config,
        "run_id": run_id, "seconds": round(time.time() - t0, 1),
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }, indent=1))
    final_dir = prompt_dir(out, pid)
    if final_dir.exists():  # left behind by hand; is_done() already handled dirs with DONE
        shutil.rmtree(final_dir)
    os.rename(tmp, final_dir)
    return final


def main():
    args = parse_args()
    set_determinism()
    out = Path(args.out)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    (out / "run_meta").mkdir(exist_ok=True)
    rows = load_geneval()
    write_text_once(out / "prompts.jsonl", "".join(json.dumps({"prompt_id": i, **r}) + "\n" for i, r in enumerate(rows)))
    write_text_once(out / "folds.json", json.dumps(make_folds(rows), indent=1) + "\n")

    config = build_config(args)
    prompt_ids = selected_prompt_ids(args)
    todo = [pid for pid in prompt_ids if not is_done(out, pid, config)]
    print(f"{len(prompt_ids) - len(todo)}/{len(prompt_ids)} prompts already done, {len(todo)} to run", flush=True)
    if not todo:
        return
    for pid in todo:
        for stale in (out / "prompts").glob(f".tmp-{pid:04d}-*"):
            shutil.rmtree(stale)

    label = f"shard{args.shard}of{args.num_shards}" if args.shard is not None else "ids"
    run_id = f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{socket.gethostname()}-{label}"
    pipe, scorers = load_sd15(DEVICE), Scorers(DEVICE)
    meta = run_meta(run_id, args, config, prompt_ids)
    meta_path = out / "run_meta" / f"{run_id}.json"
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"run {run_id} on {meta['gpu']['name']}", flush=True)

    if args.verify_pipeline:
        prompt = rows[todo[0]]["prompt"].strip()
        diff = compare_with_pipeline(pipe, prompt, initial_noise(config["seeds"], "cpu", DEVICE), config["num_steps"])
        meta["pipeline_check"] = {"prompt_id": todo[0], "max_abs_diff_final_latent": diff}
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"sampler vs PSP pipeline, prompt {todo[0]}, final latent: max |diff| = {diff:.3g}", flush=True)
        if diff > 1e-2:
            raise SystemExit("the sampler does not reproduce PSP's pipeline")

    seconds, status = [], "running"
    try:
        for i, pid in enumerate(todo, 1):
            t0 = time.time()
            final = collect_prompt(pipe, scorers, out, pid, rows[pid], config, run_id)
            seconds.append(time.time() - t0)
            eta_h = np.mean(seconds) * (len(todo) - i) / 3600
            print(f"[{i}/{len(todo)}] prompt {pid:03d} ({rows[pid]['tag']}) {seconds[-1]:.0f}s | "
                  f"final IR {final.ir.mean():.3f} HPS {final.hps.mean():.3f} ITM {final.blip_itm.mean():.3f} | "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB | ETA {eta_h:.1f} h", flush=True)
        status = "finished"
    except BaseException as exc:
        status = f"failed: {exc!r}"
        raise
    finally:
        meta.update({
            "status": status, "ended_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompts_done": len(seconds), "seconds_per_prompt": [round(s, 1) for s in seconds],
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        })
        meta_path.write_text(json.dumps(meta, indent=1))

    per_prompt = np.mean(seconds[1:] if len(seconds) > 1 else seconds)  # the first prompt includes warm-up
    print(f"{len(seconds)} prompts in {sum(seconds):.0f}s ({per_prompt:.1f} s/prompt after the first) -> "
          f"all {NUM_PROMPTS} prompts: {per_prompt * NUM_PROMPTS / 3600:.1f} h on one GPU, "
          f"{per_prompt * NUM_PROMPTS / 3600 / 4:.1f} h on four | peak allocated {meta['peak_allocated_gib']} GiB, "
          f"reserved {meta['peak_reserved_gib']} GiB", flush=True)


if __name__ == "__main__":
    main()
