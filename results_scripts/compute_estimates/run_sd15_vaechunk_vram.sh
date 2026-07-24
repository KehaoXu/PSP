#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image:${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/fkd_diffusers:${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
PROMPT_START_ID="${PROMPT_START_ID:-0}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-auto}"
RUN_LABEL="${RUN_LABEL:-vaechunk}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"

BENCH_SCRIPT="${REPO_ROOT}/results_scripts/compute_estimates/track_vram_bon_vs_pps.py"
PROMPTS_PATH="${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/prompt_files/benchmark_ir.json"

run_k() {
  local k="$1"
  local out_root="${REPO_ROOT}/results/compute_estimates/rebuttal/sd15_vaechunk_K${k}_${RUN_LABEL}_cudnn-${CUDNN_BENCHMARK}_${RUN_TAG}"
  echo "[sd15-vaechunk] K=${k} output: ${out_root}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${BENCH_SCRIPT}" \
    --prompts-path "${PROMPTS_PATH}" \
    --output-dir "${out_root}" \
    --model-name "stable-diffusion-v1-5" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --prompt-start-id "${PROMPT_START_ID}" \
    --num-prompts-total 11 \
    --num-prompts-stats 10 \
    --time-steps 64 \
    --methods "pps" \
    --pps-vae-decode-batch "${k}" \
    --cudnn-benchmark "${CUDNN_BENCHMARK}" \
    --eta 0
}

run_k 1
run_k 2
run_k 4

echo "[sd15-vaechunk] done"
