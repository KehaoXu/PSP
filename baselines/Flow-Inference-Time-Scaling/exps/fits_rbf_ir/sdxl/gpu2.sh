#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="2" PROMPT_START_ID="139" NUM_PROMPTS="69" bash "${SCRIPT_DIR}/run_sdxl_fits_rbf_shared.sh"
