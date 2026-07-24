#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

for SEED in 42 43 44; do
  export PYTHONNOUSERSITE=1
  unset PYTHONPATH || true
  CUDA_VISIBLE_DEVICES=6 python "${REPO_ROOT}/baselines/DSearch/run_dsearch_geneval.py" \
    --model-key "sd35" \
    --guidance-reward-fn "ImageReward" \
    --metrics-to-compute "ImageReward#HumanPreference" \
    --device "cuda" \
    --seed "${SEED}" \
    --prompt-start-id 415 \
    --num-prompts 69 \
    --num-particles 4 \
    --oversamplerate 2 \
    --w 2.7 \
    --search-schedule exponential \
    --drop-schedule exponential \
    --num-inference-steps 32 \
    --eta 1.0 \
    --output-root "${REPO_ROOT}/results/dsearch" \
    --prompts-path "${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl" \
    --stochastic-sampling \
    --use-step-wrapper-stochastic \
    --gamma-target 0.005
done
