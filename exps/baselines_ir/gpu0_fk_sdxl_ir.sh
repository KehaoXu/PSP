#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image"

CUDA_VISIBLE_DEVICES=0 python launch_eval_runs.py \
  --use_smc \
  --model_idx 10 \
  --output_name "outputs/fk_sdxl_k4_t64_eta1_ir_geneval" \
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
  --potential_type max

CUDA_VISIBLE_DEVICES=0 python launch_eval_runs.py \
  --use_smc \
  --model_idx 10 \
  --output_name "outputs/fk_sdxl_k4_t64_eta1_ir_benchmark_ir" \
  --prompt_path "prompt_files/benchmark_ir.json" \
  --metrics_to_compute "ImageReward#HumanPreference" \
  --guidance_reward_fn "ImageReward" \
  --num_inference_steps 64 \
  --eta 1 \
  --lmbda 10.0 \
  --resample_frequency 12 \
  --resample_t_start 12 \
  --resample_t_end 48 \
  --num_particles 4 \
  --potential_type max
