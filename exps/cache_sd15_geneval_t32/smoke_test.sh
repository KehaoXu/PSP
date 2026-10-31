#!/usr/bin/env bash
# Run-plan steps 1-2 plus two equivalence checks, before the full run. Writes to $WORKSPACE/smoke_cache.
#   1. 2 prompts x 32 seeds on GPU 0: timing (T2), peak memory, files, sampler == PSP pipeline, GPU checks
#   2. prompt 0 again on GPU 1 (GPU 0 if there is only one): everything must be bit-identical
#   3. the cache's code, given the midterm's noise and batch sizes, reproduces the midterm IR
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
SMOKE="$WORKSPACE/smoke_cache"
rm -rf "$SMOKE"
mkdir -p "$SMOKE"
GPU_B=$(( $(nvidia-smi -L | wc -l) > 1 ? 1 : 0 ))

echo "== 1. two prompts on GPU 0"
CUDA_VISIBLE_DEVICES=0 python "$HERE/collect_cache.py" --out "$SMOKE/a" --prompt-ids 0 1 --verify-pipeline \
  2>&1 | tee "$SMOKE/a.log"
CUDA_VISIBLE_DEVICES=0 python "$HERE/check_cache.py" full "$SMOKE/a" --prompt-ids 0 1 --gpu --n-rescore 4 --n-final 2

echo "== 2. prompt 0 again on GPU $GPU_B"
CUDA_VISIBLE_DEVICES=$GPU_B python "$HERE/collect_cache.py" --out "$SMOKE/b" --prompt-ids 0 2>&1 | tee "$SMOKE/b.log"
python "$HERE/check_cache.py" same "$SMOKE/a" "$SMOKE/b"

echo "== 3. midterm reproduction"
CUDA_VISIBLE_DEVICES=0 python "$HERE/verify_midterm.py"

echo "smoke test passed; see the timing line at the end of $SMOKE/a.log"
