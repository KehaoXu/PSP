#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Debug-friendly defaults for a 1xA6000 machine.
PROMPT_START_ID="${PROMPT_START_ID:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-1}"
SEED_START="${SEED_START:-0}"
NUM_SEEDS="${NUM_SEEDS:-8}"
TIME_STEPS="${TIME_STEPS:-64}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
ETA_VALUE="${ETA_VALUE:-0}"
COLLECT_HPS="${COLLECT_HPS:-1}"   # set to 0 to pass --no_hps

PROMPTS_PATH="${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/prompt_files/benchmark_ir.json"
OUT_ROOT="${REPO_ROOT}/output/sdxl_reward_signal_debug"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUT_ROOT}/benchmark_gpu${CUDA_DEVICE}_${RUN_TAG}"
LOG_PATH="${OUT_ROOT}/benchmark_gpu${CUDA_DEVICE}_${RUN_TAG}.log"

mkdir -p "${OUT_ROOT}"

{
  echo "================ SDXL DEBUG RUN ================"
  echo "timestamp: $(date -Iseconds)"
  echo "repo_root: ${REPO_ROOT}"
  echo "script: ${BASH_SOURCE[0]}"
  echo "output_dir: ${OUTPUT_DIR}"
  echo "prompt_file: ${PROMPTS_PATH}"
  echo "-------------------------------------------------"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}"
  echo "PROMPT_START_ID=${PROMPT_START_ID}"
  echo "NUM_PROMPTS=${NUM_PROMPTS}"
  echo "SEED_START=${SEED_START}"
  echo "NUM_SEEDS=${NUM_SEEDS}"
  echo "TIME_STEPS=${TIME_STEPS}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "ETA_VALUE=${ETA_VALUE}"
  echo "COLLECT_HPS=${COLLECT_HPS}"
  echo "-------------------------------------------------"
  echo "nvidia-smi:"
  nvidia-smi || true
  echo "-------------------------------------------------"
  echo "python env:"
  python - <<'PY'
import platform
import torch
import diffusers
print("python:", platform.python_version())
print("torch:", torch.__version__)
print("diffusers:", diffusers.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device_count:", torch.cuda.device_count())
    print("cuda_current_device:", torch.cuda.current_device())
    print("cuda_device_name:", torch.cuda.get_device_name(torch.cuda.current_device()))
PY
  echo "-------------------------------------------------"
  echo "prompt quick check:"
  python - <<'PY'
import json
from pathlib import Path
p = Path("Fk-Diffusion-Steering/text_to_image/prompt_files/benchmark_ir.json")
data = json.loads(p.read_text())
print("prompt_file_exists:", p.exists())
print("num_items:", len(data) if isinstance(data, list) else "non-list")
if isinstance(data, list) and data:
    first = data[0]
    if isinstance(first, dict):
        text = str(first.get("prompt", ""))
    else:
        text = str(first)
    print("first_prompt_preview:", text[:200].replace("\n", " "))
PY
  echo "================================================="
} | tee "${LOG_PATH}"

NO_HPS_FLAG=""
if [[ "${COLLECT_HPS}" == "0" ]]; then
  NO_HPS_FLAG="--no_hps"
fi

CMD=(
  python "${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/collect_image_reward_signal.py"
  --prompts-path "${PROMPTS_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --model-name "stable-diffusion-xl"
  --device "cuda"
  --seed "${SEED_START}"
  --prompt-start-id "${PROMPT_START_ID}"
  --num-prompts "${NUM_PROMPTS}"
  --num-seeds "${NUM_SEEDS}"
  --time-steps "${TIME_STEPS}"
  --batch-size "${BATCH_SIZE}"
  --eta "${ETA_VALUE}"
)
if [[ -n "${NO_HPS_FLAG}" ]]; then
  CMD+=("${NO_HPS_FLAG}")
fi

{
  echo
  echo "[debug_sdxl_collect] Running command:"
  printf ' %q' env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${CMD[@]}"
  echo
  echo
} | tee -a "${LOG_PATH}"

env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${CMD[@]}" 2>&1 | tee -a "${LOG_PATH}"

{
  echo
  echo "================ POST-RUN SUMMARY ================"
  python - "${OUTPUT_DIR}/metrics.csv" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path)
print("metrics_csv:", path)
print("rows:", len(df))
print("unique_prompts:", df["prompt_id"].nunique())
print("unique_seeds:", df["seed"].nunique())
print("step_range:", int(df["step"].min()), "to", int(df["step"].max()))

dups = int(df.duplicated(subset=["prompt_id", "seed", "step"]).sum())
print("duplicate_prompt_seed_step_rows:", dups)

final_step = int(df["step"].max())
dff = df[df["step"] == final_step].copy()
print("\nfinal_step_seed_means:")
print(
    dff.groupby("seed", as_index=False)
    .agg(
        mean_ir=("image_reward", "mean"),
        mean_hps=("human_preference", "mean"),
        prompts=("prompt_id", "nunique"),
    )
    .sort_values("seed")
    .to_string(index=False)
)
print("\nfinal_step_overall:")
print(
    dff.agg(
        mean_ir=("image_reward", "mean"),
        median_ir=("image_reward", "median"),
        mean_hps=("human_preference", "mean"),
        median_hps=("human_preference", "median"),
    ).to_string()
)
PY
  echo "=================================================="
  echo "Full log: ${LOG_PATH}"
  echo "Paste this log back for diagnosis."
} | tee -a "${LOG_PATH}"

