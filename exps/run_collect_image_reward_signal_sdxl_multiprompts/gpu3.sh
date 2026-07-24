#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CUDA_VISIBLE_DEVICES=3 python "${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/collect_image_reward_signal_multiprompts.py" \
  --prompts-path "${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata_multiprompts.json" \
  --output-dir "${REPO_ROOT}/output/sdxl_reward_signal_multiprompts_geneval_gpu3_208_276_run01" \
  --model-name "stable-diffusion-xl" \
  --device "cuda" \
  --seed 0 \
  --prompt-start-id 208 \
  --num-prompts 69 \
  --num-seeds 24 \
  --time-steps 64 \
  --batch-size 2 \
  --eta 0 \
  --no_hps
