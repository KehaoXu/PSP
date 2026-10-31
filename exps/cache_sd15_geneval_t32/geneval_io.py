#!/usr/bin/env python3
"""Hand the cache's final images to GenEval and bring the verdicts back (runs in the cache venv;
only geneval/evaluation/evaluate_images.py runs in the GenEval environment, see README.md).

  export CACHE VIEW_DIR [--copy]   VIEW_DIR/<00000>/metadata.jsonl + samples/<seed>.png -> final PNG
                                   (GenEval's folder layout; it only reads files named \\d+.png)
  merge CACHE RESULTS_JSONL        adds geneval_correct / geneval_reason to every prompts/<pid>/final.parquet,
                                   rebuilds final_all.parquet and prints the GenEval score
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from cache_lib import NUM_PROMPTS, png_name, prompt_dir

PSP_SD15_STANDARD_T64 = 0.434  # PSP README, SD1.5 Standard


def done_rows(cache):
    rows = [json.loads(l) for l in (Path(cache) / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if (prompt_dir(cache, r["prompt_id"]) / "DONE").exists()]


def cmd_export(args):
    view = Path(args.view_dir)
    n_images = 0
    for row in done_rows(args.cache):
        pid = row.pop("prompt_id")
        seeds = json.loads((prompt_dir(args.cache, pid) / "DONE").read_text())["config"]["seeds"]
        samples = view / f"{pid:05d}" / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        (samples.parent / "metadata.jsonl").write_text(json.dumps(row) + "\n")
        for seed in seeds:
            src = (prompt_dir(args.cache, pid) / "final" / png_name(seed)).resolve()
            dst = samples / f"{seed:04d}.png"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            shutil.copyfile(src, dst) if args.copy else os.symlink(src, dst)
            n_images += 1
    print(f"exported {n_images} images to {view}")


def cmd_merge(args):
    res = pd.read_json(args.results, lines=True)
    paths = res.filename.map(Path)
    res["prompt_id"] = paths.map(lambda p: int(p.parent.parent.name))
    res["seed"] = paths.map(lambda p: int(p.stem))
    res = res.rename(columns={"correct": "geneval_correct", "reason": "geneval_reason"})
    for pid, g in res.groupby("prompt_id"):
        path = prompt_dir(args.cache, pid) / "final.parquet"
        final = pd.read_parquet(path).drop(columns=["geneval_correct", "geneval_reason"], errors="ignore")
        final = final.merge(g[["seed", "geneval_correct", "geneval_reason"]], on="seed", how="left", validate="1:1")
        assert final.geneval_correct.notna().all(), f"prompt {pid}: GenEval results missing for some seeds"
        tmp = path.with_name(".final.parquet.tmp")
        final.to_parquet(tmp, index=False)
        os.replace(tmp, path)

    pids = [r["prompt_id"] for r in done_rows(args.cache)]
    final = pd.concat([pd.read_parquet(prompt_dir(args.cache, pid) / "final.parquet") for pid in pids], ignore_index=True)
    final.to_parquet(Path(args.cache) / "final_all.parquet", index=False)
    scored = final[final.geneval_correct.notna()].astype({"geneval_correct": bool})
    by_tag = scored.groupby("category").geneval_correct.mean()
    print(f"{len(scored)} images with GenEval results ({len(final)} final images, {len(pids)}/{NUM_PROMPTS} prompts)")
    print(by_tag.round(4).to_string())
    print(f"GenEval score (mean over the 6 tasks, as summary_scores.py): {by_tag.mean():.4f} "
          f"| image-level accuracy {scored.geneval_correct.mean():.4f} "
          f"| PSP SD1.5 Standard (T=64): {PSP_SD15_STANDARD_T64}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("cache")
    exp.add_argument("view_dir")
    exp.add_argument("--copy", action="store_true", help="copy the PNGs instead of symlinking them")
    mrg = sub.add_parser("merge")
    mrg.add_argument("cache")
    mrg.add_argument("results")
    args = p.parse_args()
    cmd_export(args) if args.cmd == "export" else cmd_merge(args)


if __name__ == "__main__":
    main()
