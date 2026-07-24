#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${MODEL_KEY:?MODEL_KEY not set}"
: "${GUIDANCE_REWARD_FN:?GUIDANCE_REWARD_FN not set}"
: "${GPU_ID:?GPU_ID not set}"
: "${PROMPT_START_ID:?PROMPT_START_ID not set}"
: "${NUM_PROMPTS:?NUM_PROMPTS not set}"
: "${NUM_INFERENCE_STEPS:?NUM_INFERENCE_STEPS not set}"

for SEED in 42 43 44; do
  export PYTHONNOUSERSITE=1
  unset PYTHONPATH || true
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${REPO_ROOT}/baselines/DSearch/run_dsearch_vs_geneval.py"     --model-key "${MODEL_KEY}"     --guidance-reward-fn "${GUIDANCE_REWARD_FN}"     --metrics-to-compute "${GUIDANCE_REWARD_FN}"     --device "cuda"     --seed "${SEED}"     --prompt-start-id "${PROMPT_START_ID}"     --num-prompts "${NUM_PROMPTS}"     --num_images 1     --bs 1     --duplicate_size 1     --w 2     --oversamplerate 2     --search_schudule all     --drop_schudule exponential     --replacerate 0     --variant PM     --num-inference-steps "${NUM_INFERENCE_STEPS}"     --eta 1.0     --output-root "${REPO_ROOT}/results/dsearch_v2"     --prompts-path "${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl"     --effective-c-report-path "${REPO_ROOT}/exps/dsearch_v2/effective_c_launch_report.txt"     ${EXTRA_ARGS:-}
done
