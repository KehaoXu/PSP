#!/usr/bin/env python3
"""Check that this code reproduces the midterm run (exps/run_collect_image_reward_signal_sdv15_colab).

The cache itself cannot be compared with the midterm: it uses seeds 0..31 for every prompt, drawn on
CPU generators, while the midterm (PSP's collect script) used seed = prompt_id x 16 + i on CUDA
generators. Instead, this feeds the cache's sampler, decoder and IR scorer the midterm's noise and
batch sizes (16 seeds per UNet batch, decode and score 16 at a time) and compares the IR at steps
4, 8, ..., 32 and of the final images with the midterm's ir_every4_wide.csv.
"""
import argparse
import sys

from cache_lib import REPO_ROOT, ImageRewardScorer, decode_to_pil, initial_noise, load_geneval, load_sd15, sample, set_determinism
import numpy as np
import pandas as pd

MIDTERM_CSV = REPO_ROOT / "exps" / "run_collect_image_reward_signal_sdv15_colab" / "ir_every4_wide.csv"
MIDTERM_SEEDS, MIDTERM_STEPS, MIDTERM_SCORE_EVERY = 16, 32, 4


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wide-csv", default=str(MIDTERM_CSV))
    # one prompt per GenEval tag; the midterm ran every 5th prompt
    p.add_argument("--prompt-ids", type=int, nargs="+", default=[0, 120, 220, 300, 400, 500])
    args = p.parse_args()
    set_determinism()

    wide = pd.read_csv(args.wide_csv)
    rows = load_geneval()
    pipe, ir = load_sd15("cuda"), ImageRewardScorer("cuda")
    steps = list(range(MIDTERM_SCORE_EVERY, MIDTERM_STEPS + 1, MIDTERM_SCORE_EVERY))
    diffs = []
    for pid in args.prompt_ids:
        ref = wide[wide.prompt_id == pid].sort_values("seed")
        seeds = [pid * MIDTERM_SEEDS + i for i in range(MIDTERM_SEEDS)]
        assert ref.seed.tolist() == seeds, f"prompt {pid} is not in the midterm subset (every 5th prompt)"
        prompt = rows[pid]["prompt"].strip()
        x0, xt, _ = sample(pipe, prompt, initial_noise(seeds, "psp_cuda", "cuda"), MIDTERM_STEPS,
                           xt_steps={MIDTERM_STEPS})
        got = {f"ir_x0_step{k:02d}": ir(prompt, decode_to_pil(pipe, x0[:, k - 1], MIDTERM_SEEDS)) for k in steps}
        got["ir_final"] = ir(prompt, decode_to_pil(pipe, xt[MIDTERM_STEPS], MIDTERM_SEEDS))
        d = pd.DataFrame({c: np.abs(v - ref[c].to_numpy()) for c, v in got.items()})
        print(f"prompt {pid:03d} ({rows[pid]['tag']}): |IR - midterm| max {d.values.max():.2e}, mean {d.values.mean():.2e}",
              flush=True)
        diffs.append(d)

    d = pd.concat(diffs)
    print("max |IR - midterm| per column:", d.max().map("{:.2e}".format).to_dict())
    worst, mean = d.values.max(), d.values.mean()
    if worst < 1e-3:
        print("reproduced: IR matches the midterm run")
    elif mean < 1e-2:
        print(f"close (mean |diff| {mean:.1e}, max {worst:.1e}): floating-point differences, e.g. the midterm "
              "ran on an L4; check that seeds, steps and prompts line up before relying on it")
    else:
        print(f"NOT reproduced (mean |diff| {mean:.1e}, max {worst:.1e})")
        sys.exit(1)


if __name__ == "__main__":
    main()
