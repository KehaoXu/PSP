#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${GPU_ID:?GPU_ID not set}"
: "${PROMPT_START_ID:?PROMPT_START_ID not set}"
: "${NUM_PROMPTS:?NUM_PROMPTS not set}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python "baselines/SVDD-image/run_svdd_geneval.py" \
  --model_key sd15 \
  --output_root "results/svdd" \
  --exp_name "svdd_sd15_t64_ir_geneval" \
  --prompt_path "Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl" \
  --reward imagereward \
  --guidance_reward_fn "ImageReward" \
  --num_inference_steps 64 \
  --guidance_scale 7.5 \
  --height 512 \
  --width 512 \
  --variant PM \
  --duplicate_size 4 \
  --prompt-start-id "${PROMPT_START_ID}" \
  --num-prompts "${NUM_PROMPTS}"
