#!/usr/bin/env bash
set -euo pipefail

# Run this script from geneval/ directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BESTOFN_OUTPUT_ROOT="${REPO_ROOT}/Fk-Diffusion-Steering/text_to_image/outputs/baseline_outputs/bestofn_sdv35_n4_t32_eta1_geneval"
RESULTS_ROOT="results/sdv3.5/geneval_best_of_n_output_run02_eta=0"
SUMMARY_ONLY=0

if [[ "${1:-}" == "--summary-only" ]]; then
  SUMMARY_ONLY=1
fi

for SEED in 42 43 44; do
  RUN_DIR="$(ls -dt "${BESTOFN_OUTPUT_ROOT}/seed=${SEED}_"* 2>/dev/null | head -n1)"
  if [[ -z "${RUN_DIR}" ]]; then
    echo "Could not find run folder for seed=${SEED} under ${BESTOFN_OUTPUT_ROOT}"
    exit 1
  fi

  SEED_OUT_DIR="${RESULTS_ROOT}/seed=${SEED}"
  mkdir -p "${SEED_OUT_DIR}"

  ALL_JSONL="${SEED_OUT_DIR}/results_samples.jsonl"
  SAMPLE0_JSONL="${SEED_OUT_DIR}/results_sample0.jsonl"
  BESTOFN_JSONL="${SEED_OUT_DIR}/results_best_of_n_samples.jsonl"

  if [[ "${SUMMARY_ONLY}" -eq 0 ]]; then
    echo "=== seed=${SEED}: evaluate all samples from ${RUN_DIR} ==="
    python evaluation/evaluate_images.py \
      "${RUN_DIR}" \
      --outfile "${ALL_JSONL}" \
      --samples-dir "samples" \
      --model-path "objdet"
  else
    if [[ ! -f "${ALL_JSONL}" ]]; then
      echo "Missing ${ALL_JSONL}. Run without --summary-only first."
      exit 1
    fi
  fi

  echo
  echo "=== seed=${SEED}: summary for normal inference (mean over all samples) ==="
  python evaluation/summary_scores.py "${ALL_JSONL}"

  echo
  echo "=== seed=${SEED}: summary for best_of_n_IR (sample 0 only) ==="
  python evaluation/select_one_sample_per_prompt.py \
    --input "${ALL_JSONL}" \
    --output "${SAMPLE0_JSONL}" \
    --mode sample0
  python evaluation/summary_scores.py "${SAMPLE0_JSONL}"

  echo
  echo "=== seed=${SEED}: summary for best-of-n (one best sample per prompt) ==="
  python evaluation/select_one_sample_per_prompt.py \
    --input "${ALL_JSONL}" \
    --output "${BESTOFN_JSONL}" \
    --mode bestofn
  python evaluation/summary_scores.py "${BESTOFN_JSONL}"
  echo
done
