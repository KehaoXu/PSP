#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize best strategies per setting from a folder of best-strategies JSON files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Folder containing setting JSON files (e.g. results/best_strategies/sd35).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to save summary JSON.",
    )
    return parser.parse_args()


def _load_meta(path: Path) -> dict[str, Any]:
    try:
        import ijson  # type: ignore
    except ImportError:
        ijson = None

    if ijson is None:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("meta", {})

    with path.open("rb") as f:
        return dict(ijson.kvitems(f, "meta"))


def _iter_strategies(path: Path) -> Iterable[dict[str, Any]]:
    try:
        import ijson  # type: ignore
    except ImportError:
        ijson = None

    if ijson is None:
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("strategies", []):
            yield item
        return

    with path.open("rb") as f:
        yield from ijson.items(f, "strategies.item")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _pick_best_strategy(path: Path, rank_metric: str) -> dict[str, Any]:
    best_item: dict[str, Any] | None = None
    best_rank: float | None = None
    best_score = float("-inf")

    for item in _iter_strategies(path):
        rank_val = item.get("rank", None)
        score_val = _safe_float(item.get("seed_mean", {}).get(rank_metric))

        has_rank = isinstance(rank_val, (int, float))
        if has_rank:
            rank_num = float(rank_val)
            if best_rank is None or rank_num < best_rank:
                best_item = item
                best_rank = rank_num
                best_score = score_val
            continue

        if score_val > best_score:
            best_item = item
            best_score = score_val

    if best_item is None:
        raise ValueError(f"No strategies found in {path}")
    return best_item


def summarize_file(path: Path) -> dict[str, Any]:
    meta = _load_meta(path)
    rank_metric = str(meta.get("rank_metric", "IR"))
    best = _pick_best_strategy(path, rank_metric=rank_metric)

    return {
        "setting": path.stem,
        "file": str(path),
        "model_label": meta.get("model_label"),
        "dataset_label": meta.get("dataset_label"),
        "guidance_metric": meta.get("guidance_metric"),
        "rank_metric": rank_metric,
        "best_rank": best.get("rank"),
        "best_seed_mean": best.get("seed_mean", {}),
        "best_strategy": best.get("strategy", {}),
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No .json files found in {input_dir}")

    out: list[dict[str, Any]] = []
    for path in tqdm(json_files, desc="Summarizing settings", unit="file"):
        out.append(summarize_file(path))

    print(f"Scanned {len(out)} settings from {input_dir}")
    for item in out:
        print(
            f"- {item['setting']}: rank_metric={item['rank_metric']} "
            f"best_rank={item['best_rank']} best_seed_mean={item['best_seed_mean']} "
            f"best_strategy={item['best_strategy']}"
        )

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"input_dir": str(input_dir), "num_settings": len(out), "settings": out}
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved summary JSON: {output_path}")

    # Helpful message in case ijson is missing for huge files.
    try:
        import ijson  # noqa: F401 # type: ignore
    except ImportError:
        print(
            "Note: install 'ijson' for streaming parse on very large JSON files "
            "(pip install ijson)."
        )


if __name__ == "__main__":
    main()
