#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="0" PROMPT_START_ID="0" NUM_PROMPTS="139" bash "${SCRIPT_DIR}/run_sd15_bfs_shared.sh"
