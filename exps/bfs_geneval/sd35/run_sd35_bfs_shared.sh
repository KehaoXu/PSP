#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image"

: "${GPU_ID:?GPU_ID not set}"
: "${PROMPT_START_ID:?PROMPT_START_ID not set}"
: "${NUM_PROMPTS:?NUM_PROMPTS not set}"

for SEED in 42 43 44; do
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python launch_eval_runs_sd35.py \
    --use_smc \
    --single_seed_mode \
    --seed "${SEED}" \
    --model_idx 18 \
    --model_name "stabilityai/stable-diffusion-3.5-large" \
    --output_name "${REPO_ROOT}/results/bfs/sd35/bfs_sdv35_t32_ir_geneval" \
    --prompt_path "prompt_files/geneval_metadata.jsonl" \
    --metrics_to_compute "ImageReward#HumanPreference" \
    --guidance_reward_fn "ImageReward" \
    --num_inference_steps 32 \
    --stochastic_sampling \
    --use_step_wrapper_stochastic \
    --gamma_target 0.005 \
    --lmbda 10.0 \
    --resample_frequency 6 \
    --resample_t_start 6 \
    --resample_t_end 24 \
    --num_particles 4 \
    --potential_type max \
    --resampling ssp \
    --tempering_schedule increase \
    --prompt-start-id "${PROMPT_START_ID}" \
    --num-prompts "${NUM_PROMPTS}"
done
