#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_KEY="sdxl" GUIDANCE_REWARD_FN="ImageReward" GPU_ID="5" PROMPT_START_ID="346" NUM_PROMPTS="69" NUM_INFERENCE_STEPS="64" EXTRA_ARGS="" bash "${SCRIPT_DIR}/../run_dsearch_vs_shared.sh"
