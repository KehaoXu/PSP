#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image:${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/fkd_diffusers${PYTHONPATH:+:${PYTHONPATH}}"

GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
PROMPT_START_ID="${PROMPT_START_ID:-0}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${REPO_ROOT}/results/compute_estimates/rebuttal/sd35_bon_vs_pps_ir_${RUN_TAG}"

BENCH_SCRIPT="${REPO_ROOT}/results_scripts/compute_estimates/compare_bon_vs_pps_wallclock.py"
PROMPTS_PATH="${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/prompt_files/benchmark_ir.json"

echo "[sd35-bon-vs-pps] output root: ${OUT_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${BENCH_SCRIPT}" \
  --prompts-path "${PROMPTS_PATH}" \
  --output-dir "${OUT_ROOT}" \
  --model-name "stable-diffusion-3.5-large" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --prompt-start-id "${PROMPT_START_ID}" \
  --num-prompts-total 11 \
  --num-prompts-stats 10 \
  --time-steps 32 \
  --eta 0

echo "[sd35-bon-vs-pps] done"
echo "[sd35-bon-vs-pps] results: ${OUT_ROOT}"
