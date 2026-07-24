#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

DSEARCH_ROOT="results/dsearch/sdxl/dsearch_sdxl_k4_t64_ir_geneval"
METADATA_PATH="Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl"
MODEL_PATH="geneval/objdet"

echo "[sdxl_dsearch_geneval_eval_ir_geneval] Running command:"
echo "python results_scripts/dsearch/compute_dsearch_geneval_summary.py \\"
echo "  --dsearch-root \"${DSEARCH_ROOT}\" \\"
echo "  --metadata-path \"${METADATA_PATH}\" \\"
echo "  --model-path \"${MODEL_PATH}\""

python results_scripts/dsearch/compute_dsearch_geneval_summary.py \
  --dsearch-root "${DSEARCH_ROOT}" \
  --metadata-path "${METADATA_PATH}" \
  --model-path "${MODEL_PATH}"
