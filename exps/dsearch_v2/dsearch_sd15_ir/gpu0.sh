#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_KEY="sd15" GUIDANCE_REWARD_FN="ImageReward" GPU_ID="0" PROMPT_START_ID="0" NUM_PROMPTS="70" NUM_INFERENCE_STEPS="64" EXTRA_ARGS="" bash "${SCRIPT_DIR}/../run_dsearch_vs_shared.sh"
