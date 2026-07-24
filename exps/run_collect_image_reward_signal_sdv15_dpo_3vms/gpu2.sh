#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CUDA_VISIBLE_DEVICES=2 python "${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/collect_image_reward_signal.py"   --prompts-path "${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl"   --output-dir "${REPO_ROOT}/output/sdv15_dpo_reward_signal_geneval_gpu2_139_207_run01"   --model-name "mhdang/dpo-sd1.5-text2image-v1"   --device "cuda"   --seed 0   --prompt-start-id 139   --num-prompts 69   --num-seeds 24   --time-steps 64   --batch-size 2   --eta 0   --no_hps
