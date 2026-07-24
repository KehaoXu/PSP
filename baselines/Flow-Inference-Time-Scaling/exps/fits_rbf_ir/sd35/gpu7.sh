#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="7" PROMPT_START_ID="484" NUM_PROMPTS="69" bash "${SCRIPT_DIR}/run_sd35_fits_rbf_shared.sh"
