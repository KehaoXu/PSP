#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

K_INIT=8
CUTOFF_TIMES="16,32"
REMAINING_PARTICLES="4,2"
TOTAL_STEPS=64
LOGICAL_SEEDS="0,1,2"

REWARDS_ROOT="results/reward_signal/sdxl_reward_signal_multiprompts"
GENEVAL_CSV="${REWARDS_ROOT}/sdxl_reward_signal_multiprompts_geneval_sample_scores.csv"
OUTPUT_DIR="results/multiprompt/sdxl"

echo "[sdxl_multiprompts_mt_scheduled_from_rewards] Running command:"
echo "python results_scripts/baselines/compute_mt_scheduled_from_rewards.py \\"
echo "  --model-label \"sdxl_multiprompts\" \\"
echo "  --guidance-metric \"ir\" \\"
echo "  --rewards-root \"${REWARDS_ROOT}\" \\"
echo "  --geneval-csv \"${GENEVAL_CSV}\" \\"
echo "  --k-init ${K_INIT} \\"
echo "  --cutoff-times \"${CUTOFF_TIMES}\" \\"
echo "  --remaining-particles \"${REMAINING_PARTICLES}\" \\"
echo "  --total-steps ${TOTAL_STEPS} \\"
echo "  --logical-seeds \"${LOGICAL_SEEDS}\" \\"
echo "  --output-dir \"${OUTPUT_DIR}\""

python results_scripts/baselines/compute_mt_scheduled_from_rewards.py \
  --model-label "sdxl_multiprompts" \
  --guidance-metric "ir" \
  --rewards-root "${REWARDS_ROOT}" \
  --geneval-csv "${GENEVAL_CSV}" \
  --k-init "${K_INIT}" \
  --cutoff-times "${CUTOFF_TIMES}" \
  --remaining-particles "${REMAINING_PARTICLES}" \
  --total-steps "${TOTAL_STEPS}" \
  --logical-seeds "${LOGICAL_SEEDS}" \
  --output-dir "${OUTPUT_DIR}"
