#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${GPU_ID:?GPU_ID not set}"
: "${PROMPT_START_ID:?PROMPT_START_ID not set}"
: "${NUM_PROMPTS:?NUM_PROMPTS not set}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python "baselines/SVDD-image/run_svdd_geneval.py" \
  --model_key sd35 \
  --output_root "results/svdd" \
  --exp_name "svdd_sd35_t32_ir_geneval" \
  --prompt_path "Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl" \
  --reward imagereward \
  --guidance_reward_fn "ImageReward" \
  --num_inference_steps 32 \
  --guidance_scale 7.0 \
  --height 1024 \
  --width 1024 \
  --stochastic_sampling \
  --use_step_wrapper_stochastic \
  --gamma_target 0.005 \
  --prompt-start-id "${PROMPT_START_ID}" \
  --num-prompts "${NUM_PROMPTS}"
