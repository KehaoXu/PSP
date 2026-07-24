#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image"

CUDA_VISIBLE_DEVICES=2 python launch_eval_bestofn_runs.py \
  --model_idx 18 \
  --best_of_n 4 \
  --output_name "outputs/bestofn_sdv35_n4_t32_eta1_geneval" \
  --prompt_path "prompt_files/geneval_metadata.jsonl" \
  --metrics_to_compute "ImageReward#HumanPreference" \
  --guidance_reward_fn "ImageReward" \
  --num_inference_steps 32 \
  --eta 1

CUDA_VISIBLE_DEVICES=2 python launch_eval_bestofn_runs.py \
  --model_idx 18 \
  --best_of_n 4 \
  --output_name "outputs/bestofn_sdv35_n4_t32_eta1_benchmark_ir" \
  --prompt_path "prompt_files/benchmark_ir.json" \
  --metrics_to_compute "ImageReward#HumanPreference" \
  --guidance_reward_fn "ImageReward" \
  --num_inference_steps 32 \
  --eta 1
