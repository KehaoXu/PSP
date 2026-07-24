#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

FK_ROOT="results/fk_steering/sdxl/fk_sdxl_k4_t64_eta1_ir_geneval"

python results_scripts/fk_steering/compute_fk_steering_geneval_summary.py \
  --fk-root "${FK_ROOT}" \
  --metadata-path "Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl" \
  --model-path "geneval/objdet"
