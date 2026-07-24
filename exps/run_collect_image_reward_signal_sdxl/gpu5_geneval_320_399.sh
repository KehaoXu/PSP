#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CUDA_VISIBLE_DEVICES=5 python "${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/collect_image_reward_signal.py" \
  --prompts-path "${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl" \
  --output-dir "${REPO_ROOT}/output/sdxl_reward_signal_geneval_gpu5_320_399_run01" \
  --model-name "stable-diffusion-xl" \
  --device "cuda" \
  --seed 0 \
  --prompt-start-id 320 \
  --num-prompts 80 \
  --num-seeds 32 \
  --time-steps 64 \
  --batch-size 32 \
  --eta 0
