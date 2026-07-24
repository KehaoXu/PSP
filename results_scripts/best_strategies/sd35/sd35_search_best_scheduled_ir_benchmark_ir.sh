#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

METRICS_CSV="results/reward_signal/sd35_reward_signal"
GENEVAL_CSV="results/reward_signal/sd35_reward_signal/sd35_geneval_sample_scores.csv"
OUTPUT_JSON="results/best_strategies/sd35/scheduled_ir_benchmark_ir.json"

echo "[sd35_search_best_scheduled_ir_benchmark_ir] Running command:"
echo "python results_scripts/best_strategies/search_best_scheduled_strategies_single_dataset.py \\"
echo "  --model-label \"sd35\" \\"
echo "  --dataset-label \"benchmark_ir\" \\"
echo "  --metrics-csv \"${METRICS_CSV}\" \\"
echo "  --geneval-csv \"${GENEVAL_CSV}\" \\"
echo "  --guidance-metric \"image_reward\" \\"
echo "  --seeds \"0,1\" \\"
echo "  --best-of-n 4 \\"
echo "  --max-cutoffs 4 \\"
echo "  --total-steps 32 \\"
echo "  --possible-cutoff-times \"2,4,6,8,10,12,14,16,18,20,22,24,26,28,30\" \\"
echo "  --possible-remaining-seeds \"16,12,8,4,3,2,1\" \\"
echo "  --output-json \"${OUTPUT_JSON}\""

python results_scripts/best_strategies/search_best_scheduled_strategies_single_dataset.py \
  --model-label "sd35" \
  --dataset-label "benchmark_ir" \
  --metrics-csv "${METRICS_CSV}" \
  --geneval-csv "${GENEVAL_CSV}" \
  --guidance-metric "image_reward" \
  --seeds "0,1" \
  --best-of-n 4 \
  --max-cutoffs 4 \
  --total-steps 32 \
  --possible-cutoff-times "2,4,6,8,10,12,14,16,18,20,22,24,26,28,30" \
  --possible-remaining-seeds "16,12,8,4,3,2,1" \
  --output-json "${OUTPUT_JSON}"
