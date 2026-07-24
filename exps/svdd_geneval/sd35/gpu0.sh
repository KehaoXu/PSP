#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="0" PROMPT_START_ID="0" NUM_PROMPTS="70" bash "${SCRIPT_DIR}/run_sd35_svdd_shared.sh"
