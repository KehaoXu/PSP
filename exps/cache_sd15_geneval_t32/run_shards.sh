#!/usr/bin/env bash
# Run-plan step 3: one shard per GPU, each in its own window of tmux session "cache", logging to
# $CACHE_DIR/logs/shard<k>.log. Running it again resumes: prompts with DONE are skipped.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
NUM_SHARDS="${NUM_SHARDS:-4}"
SESSION=cache

N_GPUS=$(nvidia-smi -L | wc -l)
if [ "$N_GPUS" -lt "$NUM_SHARDS" ]; then
  echo "need $NUM_SHARDS GPUs, found $N_GPUS" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists (tmux attach -t $SESSION)" >&2
  exit 1
fi
mkdir -p "$CACHE_DIR/logs"

tmux new-session -d -s "$SESSION" -n shard0
for k in $(seq 0 $((NUM_SHARDS - 1))); do
  [ "$k" -eq 0 ] || tmux new-window -t "$SESSION" -n "shard$k"
  tmux send-keys -t "$SESSION:shard$k" \
    "source '$HERE/env.sh' && CUDA_VISIBLE_DEVICES=$k python '$HERE/collect_cache.py' --out '$CACHE_DIR' --shard $k --num-shards $NUM_SHARDS 2>&1 | tee -a '$CACHE_DIR/logs/shard$k.log'" Enter
done
echo "started $NUM_SHARDS shards in tmux session '$SESSION' (tmux attach -t $SESSION; logs in $CACHE_DIR/logs)"
