#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="1" PROMPT_START_ID="70" NUM_PROMPTS="69" bash "${SCRIPT_DIR}/run_sdxl_svdd_shared.sh"
