#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

python geneval/evaluation/precompute_geneval_scores_sd35_collect_rewards.py \
  --rewards-root "output/sd3.5_collect_rewards" \
  --metadata-path "Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl" \
  --output-cache "paper_figures/cache/sd35_geneval_sample_scores.csv" \
  --model-path "geneval/objdet"
