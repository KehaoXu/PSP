#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="3" PROMPT_START_ID="208" NUM_PROMPTS="69" bash "${SCRIPT_DIR}/run_sd15_eps_greedy_shared.sh"
