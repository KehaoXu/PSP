#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image"

run_cmd() {
  local label="$1"
  shift
  echo "[$label] Running command:"
  printf ' %q' "$@"
  echo
  "$@"
}

run_cmd "sdv35-ir-benchmark_ir" \
  env CUDA_VISIBLE_DEVICES=0 python launch_eval_runs_sd35.py \
  --use_smc \
  --model_idx 18 \
  --model_name "stabilityai/stable-diffusion-3.5-large" \
  --output_name "outputs/fk_sdv35_k4_t32_stochastic_ir_benchmark_ir" \
  --prompt_path "prompt_files/benchmark_ir.json" \
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
  --potential_type max

run_cmd "sdv35-ir-geneval" \
  env CUDA_VISIBLE_DEVICES=0 python launch_eval_runs_sd35.py \
  --use_smc \
  --model_idx 18 \
  --model_name "stabilityai/stable-diffusion-3.5-large" \
  --output_name "outputs/fk_sdv35_k4_t32_stochastic_ir_geneval" \
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
  --potential_type max
