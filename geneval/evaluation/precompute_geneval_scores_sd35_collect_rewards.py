#!/usr/bin/env python3
"""
Precompute per-sample Geneval scores for SD3.5 collect-rewards outputs.

Outputs one cache CSV with:
- prompt_id, seed, tag, filename, correct
- geneval_soft_score = satisfied_requirements / total_requirements
- one per-task column task_<tag> with values 1/0/NaN
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute Geneval scores for all SD3.5 collect-rewards samples."
    )
    parser.add_argument(
        "--rewards-root",
        type=str,
        default="output/sd3.5_collect_rewards",
        help="Root containing sd35_reward_signal_geneval_* folders.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl",
        help="Geneval metadata jsonl indexed by prompt_id.",
    )
    parser.add_argument(
        "--output-cache",
        type=str,
        default="paper_figures/cache/sd35_geneval_sample_scores.csv",
        help="Output CSV cache path.",
    )
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="geneval/objdet")
    parser.add_argument("--options", nargs="*", type=str, default=[])
    args = parser.parse_args()
    args.options = dict(opt.split("=", 1) for opt in args.options)
    if args.model_config is None:
        args.model_config = os.path.join(
            os.path.dirname(__file__),
            "../mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py",
        )
    return args


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def timed(fn):
    def wrapper(*args, **kwargs):
        startt = time.time()
        result = fn(*args, **kwargs)
        endt = time.time()
        print(f"Function {fn.__name__!r} executed in {endt - startt:.3f}s", file=sys.stderr)
        return result

    return wrapper


@timed
def load_models(args):
    global zsc
    from mmdet.apis import init_detector
    import open_clip
    from clip_benchmark.metrics import zeroshot_classification as zsc_module

    zsc = zsc_module
    zsc.tqdm = lambda it, *args, **kwargs: it

    config_path = args.model_config
    object_detector_name = args.options.get(
        "model", "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco"
    )
    ckpt_path = os.path.join(args.model_path, f"{object_detector_name}.pth")
    object_detector = init_detector(config_path, ckpt_path, device=DEVICE)

    clip_arch = args.options.get("clip_model", "ViT-L-14")
    clip_model, _, transform = open_clip.create_model_and_transforms(
        clip_arch, pretrained="openai", device=DEVICE
    )
    tokenizer = open_clip.get_tokenizer(clip_arch)

    with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
        classnames = [line.strip() for line in cls_file]

    return object_detector, (clip_model, transform, tokenizer), classnames


COLORS = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
]
COLOR_CLASSIFIERS: Dict[str, torch.Tensor] = {}
zsc = None


class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects, args):
        self._image = image.convert("RGB")
        bgcolor = args.options.get("bgcolor", "#999")
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects
        self._args = args

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape)
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        if self._args.options.get("crop", "1") == "1":
            image = image.crop(box[:4])
        return (transform(image), 0)


def color_classification(image, bboxes, classname, args):
    if classname not in COLOR_CLASSIFIERS:
        COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
            clip_model,
            tokenizer,
            COLORS,
            [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                f"a photo of a {{c}} object",
            ],
            DEVICE,
        )
    clf = COLOR_CLASSIFIERS[classname]
    dataloader = torch.utils.data.DataLoader(
        ImageCrops(image, bboxes, args),
        batch_size=16,
        num_workers=4,
    )
    with torch.no_grad():
        pred, _ = zsc.run_classification(clip_model, clf, dataloader, DEVICE)
        return [COLORS[index.item()] for index in pred.argmax(1)]


def compute_iou(box_a, box_b):
    area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    i_area = area_fn(
        [
            max(box_a[0], box_b[0]),
            max(box_a[1], box_b[1]),
            min(box_a[2], box_b[2]),
            min(box_a[3], box_b[3]),
        ]
    )
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


def relative_position(obj_a, obj_b):
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised_offset = (
        np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
    )
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()
    dx, dy = revised_offset / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations


def evaluate_with_breakdown(image, objects, metadata, args) -> Tuple[bool, str, int, int]:
    """
    Same criteria as evaluate_images.py plus satisfied/total requirement counts.
    """
    correct = True
    reason: List[str] = []
    matched_groups = []

    total_requirements = len(metadata.get("include", [])) + len(metadata.get("exclude", []))
    satisfied_requirements = 0

    for req in metadata.get("include", []):
        classname = req["class"]
        matched = True
        found_objects = objects.get(classname, [])[: req["count"]]
        if len(found_objects) < req["count"]:
            correct = matched = False
            reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
        else:
            if "color" in req:
                colors = color_classification(image, found_objects, classname, args)
                if colors.count(req["color"]) < req["count"]:
                    correct = matched = False
                    reason.append(
                        f"expected {req['color']} {classname}>={req['count']}, found "
                        + f"{colors.count(req['color'])} {req['color']}; and "
                        + ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                    )
            if "position" in req and matched:
                expected_rel, target_group = req["position"]
                if matched_groups[target_group] is None:
                    correct = matched = False
                    reason.append(f"no target for {classname} to be {expected_rel}")
                else:
                    for obj in found_objects:
                        for target_obj in matched_groups[target_group]:
                            true_rels = relative_position(obj, target_obj)
                            if expected_rel not in true_rels:
                                correct = matched = False
                                reason.append(
                                    f"expected {classname} {expected_rel} target, found "
                                    + f"{' and '.join(true_rels)} target"
                                )
                                break
                        if not matched:
                            break
        if matched:
            matched_groups.append(found_objects)
            satisfied_requirements += 1
        else:
            matched_groups.append(None)

    for req in metadata.get("exclude", []):
        classname = req["class"]
        if len(objects.get(classname, [])) >= req["count"]:
            correct = False
            reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
        else:
            satisfied_requirements += 1

    return correct, "\n".join(reason), satisfied_requirements, total_requirements


def evaluate_image_with_breakdown(filepath: str, metadata: dict, args):
    from mmdet.apis import inference_detector

    result = inference_detector(object_detector, filepath)
    bbox = result[0] if isinstance(result, tuple) else result
    segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
    image = ImageOps.exif_transpose(Image.open(filepath))
    detected = {}
    confidence_threshold = THRESHOLD if metadata["tag"] != "counting" else COUNTING_THRESHOLD
    for index, classname in enumerate(classnames):
        ordering = np.argsort(bbox[index][:, 4])[::-1]
        ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]
        ordering = ordering[:MAX_OBJECTS].tolist()
        detected[classname] = []
        while ordering:
            max_obj = ordering.pop(0)
            detected[classname].append(
                (bbox[index][max_obj], None if segm is None else segm[index][max_obj])
            )
            ordering = [
                obj
                for obj in ordering
                if NMS_THRESHOLD == 1 or compute_iou(bbox[index][max_obj], bbox[index][obj]) < NMS_THRESHOLD
            ]
        if not detected[classname]:
            del detected[classname]

    correct, reason, satisfied, total = evaluate_with_breakdown(image, detected, metadata, args)
    return correct, reason, satisfied, total


def read_metadata_by_prompt(path: Path) -> List[dict]:
    metadata: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            metadata.append(json.loads(line))
    if not metadata:
        raise ValueError(f"No prompt metadata found in {path}")
    return metadata


def discover_jobs(rewards_root: Path, metadata: List[dict]) -> List[Tuple[int, int, str, str]]:
    """
    Returns tuples of:
      (prompt_id, seed, image_path, tag)
    """
    jobs: List[Tuple[int, int, str, str]] = []
    shard_prefixes = {rewards_root.name}
    # Backward compatibility: some SD1.5 reward folders were produced with "sdv15" prefix.
    if "sd15" in rewards_root.name:
        shard_prefixes.add(rewards_root.name.replace("sd15", "sdv15"))
    if "sdv15" in rewards_root.name:
        shard_prefixes.add(rewards_root.name.replace("sdv15", "sd15"))

    shard_dirs: List[Path] = []
    searched_globs: List[str] = []
    for prefix in sorted(shard_prefixes):
        shard_glob = f"{prefix}_geneval_*"
        searched_globs.append(shard_glob)
        shard_dirs.extend(sorted(rewards_root.glob(shard_glob)))
    shard_dirs = sorted(set(shard_dirs))
    if not shard_dirs:
        joined_globs = ", ".join(searched_globs)
        raise ValueError(f"No shard folders found under {rewards_root}. Tried: {joined_globs}")

    for shard_dir in shard_dirs:
        samples_root = shard_dir / "samples"
        if not samples_root.is_dir():
            continue
        for prompt_dir_name in sorted(os.listdir(samples_root)):
            prompt_dir = samples_root / prompt_dir_name
            if not prompt_dir.is_dir() or not prompt_dir_name.isdigit():
                continue
            prompt_id = int(prompt_dir_name)
            if prompt_id < 0 or prompt_id >= len(metadata):
                continue
            tag = str(metadata[prompt_id].get("tag", "unknown"))
            for image_name in os.listdir(prompt_dir):
                image_path = prompt_dir / image_name
                if not image_path.is_file() or not re.match(r"\d+\.png$", image_name):
                    continue
                seed = int(image_name.replace(".png", ""))
                jobs.append((prompt_id, seed, str(image_path), tag))
    if not jobs:
        raise ValueError("No sample images found for evaluation.")
    return jobs


def main(args):
    repo_root = Path(__file__).resolve().parents[2]

    rewards_root = (repo_root / args.rewards_root).resolve()
    metadata_path = (repo_root / args.metadata_path).resolve()
    output_cache = (repo_root / args.output_cache).resolve()

    # Preserve behavior from evaluate_images.py when called from geneval/.
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        args.model_path = str((repo_root / model_path).resolve())

    metadata = read_metadata_by_prompt(metadata_path)
    all_tags = sorted({str(item.get("tag", "unknown")) for item in metadata})
    jobs = discover_jobs(rewards_root, metadata)

    rows = []
    for prompt_id, seed, image_path, tag in tqdm(jobs, desc="Precomputing Geneval scores"):
        meta = metadata[prompt_id]
        correct, reason, satisfied, total = evaluate_image_with_breakdown(image_path, meta, args)
        soft_score = float(satisfied) / float(total) if total > 0 else float("nan")
        row = {
            "prompt_id": int(prompt_id),
            "seed": int(seed),
            "filename": image_path,
            "tag": tag,
            "correct": int(bool(correct)),
            "geneval_soft_score": soft_score,
            "satisfied_requirements": int(satisfied),
            "total_requirements": int(total),
        }
        for task_tag in all_tags:
            col = f"task_{task_tag}"
            row[col] = float("nan")
        row[f"task_{tag}"] = 1.0 if correct else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_cache, index=False)
    print(f"Saved cache: {output_cache}")
    print(f"Rows: {len(df)}, prompts: {df['prompt_id'].nunique()}, seeds: {df['seed'].nunique()}")


if __name__ == "__main__":
    args = parse_args()
    if DEVICE != "cuda":
        raise RuntimeError("CUDA is required to run precompute_geneval_scores_sd35_collect_rewards.py")
    object_detector, (clip_model, transform, tokenizer), classnames = load_models(args)
    THRESHOLD = float(args.options.get("threshold", 0.3))
    COUNTING_THRESHOLD = float(args.options.get("counting_threshold", 0.9))
    MAX_OBJECTS = int(args.options.get("max_objects", 16))
    NMS_THRESHOLD = float(args.options.get("max_overlap", 1.0))
    POSITION_THRESHOLD = float(args.options.get("position_threshold", 0.1))
    main(args)
