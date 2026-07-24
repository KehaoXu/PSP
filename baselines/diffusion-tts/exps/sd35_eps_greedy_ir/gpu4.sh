#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="4" PROMPT_START_ID="277" NUM_PROMPTS="69" bash "${SCRIPT_DIR}/run_sd35_eps_greedy_shared.sh"
