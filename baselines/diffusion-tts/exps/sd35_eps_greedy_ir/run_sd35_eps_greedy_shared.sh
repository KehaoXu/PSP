#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)"

: "${GPU_ID:?GPU_ID not set}"
: "${PROMPT_START_ID:?PROMPT_START_ID not set}"
: "${NUM_PROMPTS:?NUM_PROMPTS not set}"

for SEED in 42 43 44; do
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${REPO_ROOT}/run_multi_backbone_eps_greedy_geneval.py" \
    --model-key "sd35" \
    --device "cuda" \
    --seed "${SEED}" \
    --prompt-start-id "${PROMPT_START_ID}" \
    --num-prompts "${NUM_PROMPTS}" \
    --num-inference-steps 32 \
    --guidance-scale 7.0 \
    --method "eps_greedy" \
    --guidance-reward-fn "ImageReward" \
    --metrics-to-compute "ImageReward" \
    --num-images 1 \
    --bs 1 \
    --N 2 \
    --lambda_ 0.15 \
    --eps 0.4 \
    --K 2 \
    --B 2 \
    --S 8 \
    --eta 1.0 \
    --stochastic-sampling \
    --use-step-wrapper-stochastic \
    --gamma-target 0.005 \
    --output-root "${WORKSPACE_ROOT}/results/diffusion_tts_v2" \
    --prompts-path "${WORKSPACE_ROOT}/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl"
done
