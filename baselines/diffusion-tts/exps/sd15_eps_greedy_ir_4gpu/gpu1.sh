#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="1" PROMPT_START_ID="139" NUM_PROMPTS="138" bash "${SCRIPT_DIR}/run_sd15_eps_greedy_shared.sh"
