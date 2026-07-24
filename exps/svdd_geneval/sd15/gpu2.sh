#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="2" PROMPT_START_ID="277" NUM_PROMPTS="138" bash "${SCRIPT_DIR}/run_sd15_svdd_shared.sh"
