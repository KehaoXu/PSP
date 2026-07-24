#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_KEY="sd35" GUIDANCE_REWARD_FN="ImageReward" GPU_ID="0" PROMPT_START_ID="0" NUM_PROMPTS="70" NUM_INFERENCE_STEPS="32" EXTRA_ARGS="--stochastic-sampling --use-step-wrapper-stochastic --gamma-target 0.005 --resume-skip-existing" bash "${SCRIPT_DIR}/../run_dsearch_vs_shared.sh"
