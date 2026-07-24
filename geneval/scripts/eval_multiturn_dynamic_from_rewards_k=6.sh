#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?Usage: bash scripts/eval_multiturn_dynamic_from_rewards.sh <seed>}"

python evaluation/evaluate_multiturn_from_rewards.py \
  --search-mode dynamic \
  --k-init 6 \
  --cutoff-times "16,24,30" \
  --eps-list "0.25,0.25,0.1" \
  --budget 128 \
  --total-steps 32 \
  --seed "${SEED}" \
  --results-root "geneval/results/sdv3.5_hyper_search_k=6" \
  --model-path "geneval/objdet"
