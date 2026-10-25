#!/usr/bin/env bash
# Shard inference over N local GPUs (one process per GPU, sharded by video), then score once.
#   scripts/infer_multi_gpu.sh vstar_qwen3_5_4b 4 [overrides...]
set -euo pipefail
EXP=${1:?experiment}; NGPU=${2:?num_gpus}; shift 2

for ((i = 0; i < NGPU; i++)); do
  CUDA_VISIBLE_DEVICES=$i uv run stvg infer -c "$EXP" inference.shard="$i/$NGPU" "$@" &
done
wait
uv run stvg score -c "$EXP" "$@"
