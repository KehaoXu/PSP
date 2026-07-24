#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CUDA_VISIBLE_DEVICES=6 python "${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/collect_image_reward_signal.py"   --prompts-path "${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl"   --output-dir "${REPO_ROOT}/output/sdxl_dpo_reward_signal_geneval_gpu6_415_483_run01"   --model-name "mhdang/dpo-sdxl-text2image-v1"   --device "cuda"   --seed 0   --prompt-start-id 415   --num-prompts 69   --num-seeds 24   --time-steps 64   --batch-size 2   --eta 0   --no_hps
