#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

N=4
LOGICAL_SEEDS="0,1,2"
REWARDS_ROOT="results/reward_signal/sd15_reward_signal_multiprompts"
GENEVAL_CSV="${REWARDS_ROOT}/sdv15_reward_signal_multiprompts_geneval_sample_scores.csv"
OUTPUT_DIR="results/multiprompt/sdv15"

echo "[sdv15_multiprompts_regular_vs_bestofn] Running command:"
echo "python results_scripts/baselines/compute_regular_vs_bestofn.py \\"
echo "  --model-label \"sdv15_multiprompts\" \\"
echo "  --guidance-metric \"ir\" \\"
echo "  --rewards-root \"${REWARDS_ROOT}\" \\"
echo "  --geneval-csv \"${GENEVAL_CSV}\" \\"
echo "  --n ${N} \\"
echo "  --logical-seeds \"${LOGICAL_SEEDS}\" \\"
echo "  --output-dir \"${OUTPUT_DIR}\""

python results_scripts/baselines/compute_regular_vs_bestofn.py \
  --model-label "sdv15_multiprompts" \
  --guidance-metric "ir" \
  --rewards-root "${REWARDS_ROOT}" \
  --geneval-csv "${GENEVAL_CSV}" \
  --n "${N}" \
  --logical-seeds "${LOGICAL_SEEDS}" \
  --output-dir "${OUTPUT_DIR}"
