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
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-auto}"
RUN_LABEL="${RUN_LABEL:-bonn8}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${REPO_ROOT}/results/compute_estimates/rebuttal/sd35_bon_n8_wallclock_${RUN_LABEL}_cudnn-${CUDNN_BENCHMARK}_${RUN_TAG}"

BENCH_SCRIPT="${REPO_ROOT}/results_scripts/compute_estimates/compare_bon_vs_pps_wallclock.py"
PROMPTS_PATH="${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/prompt_files/benchmark_ir.json"

echo "[sd35-bon-n8] output root: ${OUT_ROOT}"
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
  --methods "bon" \
  --bon-init-particles 8 \
  --cudnn-benchmark "${CUDNN_BENCHMARK}" \
  --eta 0

echo "[sd35-bon-n8] done"
echo "[sd35-bon-n8] results: ${OUT_ROOT}"
