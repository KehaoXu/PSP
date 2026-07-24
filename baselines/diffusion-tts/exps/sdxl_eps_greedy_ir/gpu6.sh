#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="6" PROMPT_START_ID="415" NUM_PROMPTS="69" bash "${SCRIPT_DIR}/run_sdxl_eps_greedy_shared.sh"
