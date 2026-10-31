#!/usr/bin/env python3
"""Checks on the cache.

  full CACHE [--prompt-ids ...] [--gpu]   the post-run checklist (README.md); also merges the per-prompt
                                          tables into CACHE/scores_all.parquet and CACHE/final_all.parquet
                                          and writes CACHE/checks.json. Exit code 1 if a check fails.
  same CACHE_A CACHE_B                    the prompts finished in both must be bit-identical (determinism)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from cache_lib import (
    EMB_DIMS, LATENT_SHAPE, NUM_PROMPTS, SCORE_COLS, Scorers, decode_to_pil, initial_noise, load_geneval,
    load_sd15, png_name, prompt_dir,
)
import numpy as np
import pandas as pd
import torch
from PIL import Image

DEVICE = "cuda"
RESCORE_TOL = 1e-3


class Report:
    def __init__(self):
        self.items = []

    def check(self, name, ok, detail=""):
        self.items.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail != "" else ""), flush=True)

    def info(self, name, detail):
        self.items.append({"check": name, "ok": None, "detail": detail})
        print(f"[info] {name}: {json.dumps(detail)}", flush=True)

    def exit(self, path=None):
        failed = [i["check"] for i in self.items if i["ok"] is False]
        if path:
            path.write_text(json.dumps(self.items, indent=1, default=str))
        print(f"{len(failed)} failed" + (f": {failed}" if failed else ""), flush=True)
        sys.exit(1 if failed else 0)


def abs_stats(diff):
    a = np.abs(np.asarray(diff, dtype=np.float64))
    return {"mean": round(a.mean(), 4), "p50": round(np.median(a), 4), "p95": round(np.quantile(a, 0.95), 4),
            "max": round(a.max(), 4)}


def cmd_full(args):
    cache = Path(args.cache)
    rows = load_geneval()
    ids = args.prompt_ids or list(range(NUM_PROMPTS))
    rep = Report()

    missing = [pid for pid in ids if not (prompt_dir(cache, pid) / "DONE").exists()]
    rep.check("every prompt has DONE", not missing,
              f"{len(ids) - len(missing)}/{len(ids)}" + (f", missing {missing[:10]}" if missing else ""))
    if missing:
        rep.exit()
    done = {pid: json.loads((prompt_dir(cache, pid) / "DONE").read_text()) for pid in ids}
    rep.check("one config for all prompts", len({json.dumps(d["config"], sort_keys=True) for d in done.values()}) == 1)
    cfg = done[ids[0]]["config"]
    seeds, n_steps, score_steps = cfg["seeds"], cfg["num_steps"], cfg["score_steps"]
    n_seeds, n_scored = len(seeds), len(score_steps)

    # files, shapes, dtypes (array headers only)
    expect = {
        "xT.npy": (n_seeds, *LATENT_SHAPE),
        "x0_latents.npy": (n_seeds, len(cfg["x0_steps"]), *LATENT_SHAPE),
        "xt_latents.npy": (n_seeds, len(cfg["xt_steps"]), *LATENT_SHAPE),
        **{f"{name}.npy": (n_seeds, n_scored, dim) for name, dim in EMB_DIMS.items()},
    }
    bad = []
    for pid in ids:
        d = prompt_dir(cache, pid)
        for name, shape in expect.items():
            a = np.load(d / name, mmap_mode="r")
            if a.shape != shape or a.dtype != np.float16:
                bad.append(f"{pid}/{name} {a.shape} {a.dtype}")
        if sorted(p.name for p in (d / "final").glob("*.png")) != sorted(png_name(s) for s in seeds):
            bad.append(f"{pid}/final")
    rep.check("files, array shapes, fp16", not bad, "; ".join(bad[:10]))

    # merged tables
    scores = pd.concat([pd.read_parquet(prompt_dir(cache, pid) / "scores.parquet") for pid in ids], ignore_index=True)
    final = pd.concat([pd.read_parquet(prompt_dir(cache, pid) / "final.parquet") for pid in ids], ignore_index=True)
    rep.check("scores rows = prompts x seeds x scored steps", len(scores) == len(ids) * n_seeds * n_scored,
              f"{len(scores)} = {len(ids)} x {n_seeds} x {n_scored}?")
    rep.check("final rows = prompts x seeds", len(final) == len(ids) * n_seeds, f"{len(final)}")
    rep.check("no nulls", not scores.isna().any().any() and not final[SCORE_COLS].isna().any().any())
    rep.check("no duplicate keys", not scores.duplicated(["prompt_id", "seed", "step"]).any()
              and not final.duplicated(["prompt_id", "seed"]).any())
    rep.check("scored steps", sorted(scores.step.unique().tolist()) == score_steps)
    tags = {pid: rows[pid]["tag"] for pid in ids}
    rep.check("category = GenEval tag", (scores.category == scores.prompt_id.map(tags)).all()
              and (final.category == final.prompt_id.map(tags)).all())
    in_range = all(((df.blip_itm >= 0) & (df.blip_itm <= 1) & (df.blip_itc.abs() <= 1) & (df.hps.abs() <= 1)
                    & np.isfinite(df.ir)).all() for df in (scores, final))
    rep.check("scores within their ranges (ITM in [0,1], ITC and HPS cosines in [-1,1], IR finite)", in_range)
    scores.to_parquet(cache / "scores_all.parquet", index=False)
    final.to_parquet(cache / "final_all.parquet", index=False)

    # seed convention: one x_T for all prompts, equal to the documented per-seed CPU generator
    digests = {hashlib.md5(np.load(prompt_dir(cache, pid) / "xT.npy").tobytes()).hexdigest() for pid in ids}
    rep.check("x_T shared by all prompts (common random numbers)", len(digests) == 1)
    rep.check("x_T = per-seed CPU generator",
              np.array_equal(np.load(prompt_dir(cache, ids[0]) / "xT.npy"), initial_noise(seeds, "cpu", "cpu").numpy()))

    # the final latent x_t[T] = sqrt(a0) x0[T] + sqrt(1 - a0) eps with a0 = alphas_cumprod[0] (set_alpha_to_one
    # is False for SD1.5), so step-T scores are close to, not equal to, the final-image scores
    last = scores[scores.step == n_steps].merge(final, on=["prompt_id", "seed"], suffixes=("", "_final"))
    rep.info(f"|step {n_steps} x0 score - final image score|", {c: abs_stats(last[c] - last[f"{c}_final"]) for c in SCORE_COLS})

    # one GPU model and software stack across the runs that wrote these prompts
    run_ids = sorted({d["run_id"] for d in done.values()})
    meta_paths = [cache / "run_meta" / f"{r}.json" for r in run_ids]
    rep.check("run_meta present for every run", all(p.exists() for p in meta_paths), f"{len(run_ids)} runs")
    metas = [json.loads(p.read_text()) for p in meta_paths if p.exists()]
    for key, get in (("GPU model", lambda m: m["gpu"]["name"]), ("versions", lambda m: m["versions"]),
                     ("pip freeze", lambda m: m["pip_freeze"]), ("code md5", lambda m: m["code_md5"])):
        values = {json.dumps(get(m), sort_keys=True) for m in metas}
        rep.check(f"same {key} in all runs", len(values) == 1, "" if len(values) == 1 else sorted(values)[:4])

    first = scores[scores.step == score_steps[0]]
    rep.info("score summary", {
        name: df[SCORE_COLS].agg(["mean", "std", "min", "max"]).round(4).to_dict()
        for name, df in ((f"step {score_steps[0]}", first), (f"step {n_steps}", last), ("final", final))
    })
    sat = (final.blip_itm > 0.99).groupby(final.category).mean().round(3).to_dict()
    rep.info("share of final images with BLIP-ITM > 0.99", {"all": round((final.blip_itm > 0.99).mean(), 3), **sat})

    if args.gpu:
        gpu_checks(rep, cache, ids, rows, cfg, scores, final, args)
    rep.exit(cache / "checks.json")


def gpu_checks(rep, cache, ids, rows, cfg, scores, final, args):
    pipe, scorers = load_sd15(DEVICE), Scorers(DEVICE)
    rng = np.random.default_rng(args.seed)
    n_steps, seeds = cfg["num_steps"], cfg["seeds"]

    def rescore(prompt, images):
        return scorers(prompt, images, cfg["score_batch"])

    def to_gpu(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE)

    # random (prompt, scored step): re-decode that step's x0 for all seeds in the original decode batches
    by_key = scores.set_index(["prompt_id", "step"]).sort_index()
    worst = dict.fromkeys(SCORE_COLS, 0.0)
    for _ in range(args.n_rescore):
        pid, k = int(rng.choice(ids)), int(rng.choice(cfg["score_steps"]))
        x0 = np.load(prompt_dir(cache, pid) / "x0_latents.npy", mmap_mode="r")[:, cfg["x0_steps"].index(k)]
        got = rescore(rows[pid]["prompt"].strip(), decode_to_pil(pipe, to_gpu(x0), cfg["decode_batch"]))
        ref = by_key.loc[(pid, k)].sort_values("seed")
        for c in SCORE_COLS:
            worst[c] = max(worst[c], float(np.abs(got[c] - ref[c].to_numpy()).max()))
    rep.check(f"{args.n_rescore} random (prompt, step) x0 re-decoded and re-scored = scores table (< {RESCORE_TOL})",
              max(worst.values()) < RESCORE_TOL, {c: f"{v:.2e}" for c, v in worst.items()})

    # x_t[T] decodes to the saved PNGs, and re-scoring the PNGs reproduces final.parquet
    pixel_bad, worst = [], dict.fromkeys(SCORE_COLS, 0.0)
    for pid in rng.choice(ids, size=min(args.n_final, len(ids)), replace=False).tolist():
        d = prompt_dir(cache, pid)
        xt = np.load(d / "xt_latents.npy", mmap_mode="r")[:, cfg["xt_steps"].index(n_steps)]
        decoded = decode_to_pil(pipe, to_gpu(xt), cfg["decode_batch"])
        pngs = [Image.open(d / "final" / png_name(s)).convert("RGB") for s in seeds]
        pixel_bad += [f"{pid}/{s}" for s, a, b in zip(seeds, decoded, pngs) if not np.array_equal(np.asarray(a), np.asarray(b))]
        got = rescore(rows[pid]["prompt"].strip(), pngs)
        ref = final[final.prompt_id == pid].sort_values("seed")
        for c in SCORE_COLS:
            worst[c] = max(worst[c], float(np.abs(got[c] - ref[c].to_numpy()).max()))
    rep.check(f"x_t at step {n_steps} decodes to the saved PNGs ({args.n_final} prompts)", not pixel_bad, pixel_bad[:10])
    rep.check(f"re-scoring the PNGs = final table (< {RESCORE_TOL})", max(worst.values()) < RESCORE_TOL,
              {c: f"{v:.2e}" for c, v in worst.items()})


def cmd_same(args):
    a, b = Path(args.cache_a), Path(args.cache_b)
    common = sorted(p.name for p in (b / "prompts").iterdir()
                    if (p / "DONE").exists() and (a / "prompts" / p.name / "DONE").exists())
    rep = Report()
    rep.check("prompts finished in both", bool(common), common)
    for name in common:
        da, db = a / "prompts" / name, b / "prompts" / name
        diffs = []
        for f in sorted(p.name for p in da.glob("*.npy")):
            xa, xb = np.load(da / f), np.load(db / f)
            if xa.tobytes() != xb.tobytes():
                diffs.append(f"{f} (max |diff| {np.abs(xa.astype(np.float32) - xb.astype(np.float32)).max():.2e})")
        diffs += [f for f in ("scores.parquet", "final.parquet") if not pd.read_parquet(da / f).equals(pd.read_parquet(db / f))]
        diffs += [f"final/{p.name}" for p in sorted((da / "final").glob("*.png"))
                  if not np.array_equal(np.asarray(Image.open(p)), np.asarray(Image.open(db / "final" / p.name)))]
        rep.check(f"prompt {name}: latents, scores and PNGs bit-identical", not diffs, ", ".join(diffs[:8]))
    rep.exit()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    full = sub.add_parser("full")
    full.add_argument("cache")
    full.add_argument("--prompt-ids", type=int, nargs="+")
    full.add_argument("--gpu", action="store_true", help="also run the decode / re-score checks")
    full.add_argument("--n-rescore", type=int, default=20, help="random (prompt, step) pairs whose x0 is re-scored")
    full.add_argument("--n-final", type=int, default=20, help="random prompts whose final images are re-checked")
    full.add_argument("--seed", type=int, default=0)
    same = sub.add_parser("same")
    same.add_argument("cache_a")
    same.add_argument("cache_b")
    args = p.parse_args()
    cmd_full(args) if args.cmd == "full" else cmd_same(args)


if __name__ == "__main__":
    main()
