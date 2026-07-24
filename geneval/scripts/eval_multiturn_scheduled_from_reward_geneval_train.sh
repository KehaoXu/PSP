#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?Usage: bash scripts/eval_multiturn_scheduled_from_rewards.sh <seed>}"

python evaluation/evaluate_multiturn_from_rewards.py \
  --search-mode scheduled \
  --k-init 8 \
  --cutoff-times "10,12,18,26" \
  --remaining-particles "4,3,2,1" \
  --total-steps 32 \
  --seed "${SEED}" \
  --results-root "geneval/results/sdv3.5_hyper_search_geneval_train" \
  --model-path "geneval/objdet"
