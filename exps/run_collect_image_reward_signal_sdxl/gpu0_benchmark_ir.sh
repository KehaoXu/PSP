#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CUDA_VISIBLE_DEVICES=0 python "${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/collect_image_reward_signal.py" \
  --prompts-path "${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/prompt_files/benchmark_ir.json" \
  --output-dir "${REPO_ROOT}/output/sdxl_reward_signal_benchmark_gpu0_run01" \
  --model-name "stable-diffusion-xl" \
  --device "cuda" \
  --seed 0 \
  --num-seeds 32 \
  --time-steps 64 \
  --batch-size 2 \
  --eta 0
