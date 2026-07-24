#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)"

: "${GPU_ID:?GPU_ID not set}"
: "${PROMPT_START_ID:?PROMPT_START_ID not set}"
: "${NUM_PROMPTS:?NUM_PROMPTS not set}"

for SEED in 42 43 44; do
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${REPO_ROOT}/run_multi_backbone_fits_geneval.py" \
    --model-key "sdxl" \
    --device "cuda" \
    --seed "${SEED}" \
    --prompt-start-id "${PROMPT_START_ID}" \
    --num-prompts "${NUM_PROMPTS}" \
    --num-inference-steps 64 \
    --guidance-scale 7.5 \
    --image-size 1024 \
    --batch-size 2 \
    --init-n-particles 8 \
    --max-nfe 256 \
    --filtering-method "rbf" \
    --reward-score "imagereward" \
    --ckpt-root "${REPO_ROOT}/ckpt" \
    --output-root "${WORKSPACE_ROOT}/results/fits_rbf_v1" \
    --prompts-path "${WORKSPACE_ROOT}/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl"
done
