#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image"

: "${GPU_ID:?GPU_ID not set}"
: "${PROMPT_START_ID:?PROMPT_START_ID not set}"
: "${NUM_PROMPTS:?NUM_PROMPTS not set}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python launch_eval_runs.py \
  --use_smc \
  --model_idx 10 \
  --output_name "${REPO_ROOT}/results/bfs/sdxl/bfs_sdxl_t64_ir_geneval" \
  --prompt_path "prompt_files/geneval_metadata.jsonl" \
  --metrics_to_compute "ImageReward#HumanPreference" \
  --guidance_reward_fn "ImageReward" \
  --num_inference_steps 64 \
  --eta 1 \
  --lmbda 10.0 \
  --resample_frequency 12 \
  --resample_t_start 12 \
  --resample_t_end 48 \
  --num_particles 4 \
  --potential_type max \
  --resampling ssp \
  --tempering_schedule increase \
  --prompt-start-id "${PROMPT_START_ID}" \
  --num-prompts "${NUM_PROMPTS}"
