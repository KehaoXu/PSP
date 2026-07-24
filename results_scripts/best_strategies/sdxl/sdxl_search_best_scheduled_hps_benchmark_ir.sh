#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

METRICS_CSV="results/reward_signal/sdxl_reward_signal/sdxl_reward_signal_benchmark_gpu0_run01/metrics.csv"
GENEVAL_CSV="results/reward_signal/sdxl_reward_signal/sdxl_geneval_sample_scores.csv"
OUTPUT_JSON="results/best_strategies/sdxl/scheduled_hps_benchmark_ir.json"

echo "[sdxl_search_best_scheduled_hps_benchmark_ir] Running command:"
echo "python results_scripts/best_strategies/search_best_scheduled_strategies_single_dataset.py \\"
echo "  --model-label \"sdxl\" \\"
echo "  --dataset-label \"benchmark_ir\" \\"
echo "  --metrics-csv \"${METRICS_CSV}\" \\"
echo "  --geneval-csv \"${GENEVAL_CSV}\" \\"
echo "  --guidance-metric \"human_preference\" \\"
echo "  --seeds \"0,1\" \\"
echo "  --best-of-n 4 \\"
echo "  --max-cutoffs 4 \\"
echo "  --total-steps 64 \\"
echo "  --possible-cutoff-times \"4,8,12,16,20,24,28,32,36,40,44,48,52,56,60\" \\"
echo "  --possible-remaining-seeds \"16,12,8,4,3,2,1\" \\"
echo "  --output-json \"${OUTPUT_JSON}\""

python results_scripts/best_strategies/search_best_scheduled_strategies_single_dataset.py \
  --model-label "sdxl" \
  --dataset-label "benchmark_ir" \
  --metrics-csv "${METRICS_CSV}" \
  --geneval-csv "${GENEVAL_CSV}" \
  --guidance-metric "human_preference" \
  --seeds "0,1" \
  --best-of-n 4 \
  --max-cutoffs 4 \
  --total-steps 64 \
  --possible-cutoff-times "4,8,12,16,20,24,28,32,36,40,44,48,52,56,60" \
  --possible-remaining-seeds "16,12,8,4,3,2,1" \
  --output-json "${OUTPUT_JSON}"
