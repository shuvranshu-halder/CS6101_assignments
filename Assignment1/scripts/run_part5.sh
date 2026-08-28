#!/bin/bash
# Runs Part 5 (SPLADE) for one dataset.
#
# Usage:
#   bash scripts/run_part5.sh <dataset> [options]
#
# Examples:
#   bash scripts/run_part5.sh fever --stage encode
#   bash scripts/run_part5.sh fever --stage retrieve compare
#   bash scripts/run_part5.sh fever --stage encode retrieve compare
#
# Encoding is sharded across the configured GPUs. Retrieval/compare reuse
# the already-encoded SPLADE shards and run only after encoding when requested.

set -e

DATASET=$1
shift || true

if [ -z "$DATASET" ]; then
  echo "Usage: bash scripts/run_part5.sh <scifact|fever|hotpotqa|msmarco> [options]"
  exit 1
fi

# ---------------------------------------------------------------------------
# GPU CONFIG — edit per dataset.
# ---------------------------------------------------------------------------
case "$DATASET" in
  scifact)
    GPU_IDS=(2)
    BATCH_SIZE=128
    ;;
  fever|hotpotqa)
    GPU_IDS=(2 3)
    BATCH_SIZE=64
    ;;
  msmarco)
    GPU_IDS=(0 1 2 3 4 5 6 7)
    BATCH_SIZE=64
    ;;
  *)
    GPU_IDS=(0)
    BATCH_SIZE=64
    ;;
esac

export OPENAI_API_KEY="not-needed"
CHECKPOINT="naver/splade-v3"
NUM_SHARDS=${#GPU_IDS[@]}

cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Parse options.
# ---------------------------------------------------------------------------
STAGES=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        STAGES+=("$1")
        shift
      done
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      EXTRA_ARGS+=(--batch_size "$2")
      shift 2
      ;;
    --index_threads)
      EXTRA_ARGS+=(--index_threads "$2")
      shift 2
      ;;
    --query_batch_size)
      EXTRA_ARGS+=(--query_batch_size "$2")
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# Default: run all stages.
if [ ${#STAGES[@]} -eq 0 ]; then
  STAGES=(encode retrieve compare)
fi

echo "=== [$DATASET] requested stages: ${STAGES[*]} ==="

# ---------------------------------------------------------------------------
# Run corpus encoding only if explicitly requested.
# ---------------------------------------------------------------------------
if [[ " ${STAGES[*]} " == *" encode "* ]]; then
  echo "=== [$DATASET] encoding corpus across ${NUM_SHARDS} GPU(s): ${GPU_IDS[*]} ==="

  PIDS=()
  for shard_id in "${!GPU_IDS[@]}"; do
    gpu="${GPU_IDS[$shard_id]}"
    echo "  shard $shard_id -> GPU $gpu"

    CUDA_VISIBLE_DEVICES="$gpu" python -u run_part5.py \
      --dataset "$DATASET" \
      --checkpoint "$CHECKPOINT" \
      --stage encode \
      --shard_id "$shard_id" \
      --num_shards "$NUM_SHARDS" \
      --batch_size "$BATCH_SIZE" \
      > "logs/${DATASET}_splade_shard${shard_id}.log" 2>&1 &

    PIDS+=($!)
  done

  echo "=== waiting for all shards to finish encoding ==="
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done

  echo "=== [$DATASET] corpus encoding complete ==="
fi

# ---------------------------------------------------------------------------
# Retrieval.
# ---------------------------------------------------------------------------
if [[ " ${STAGES[*]} " == *" retrieve "* ]]; then
  echo "=== [$DATASET] SPLADE retrieval ==="

  CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" python -u run_part5.py \
    --dataset "$DATASET" \
    --checkpoint "$CHECKPOINT" \
    --stage retrieve \
    "${EXTRA_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Expansion-term comparison.
# ---------------------------------------------------------------------------
if [[ " ${STAGES[*]} " == *" compare "* ]]; then
  echo "=== [$DATASET] expansion-term comparison ==="

  CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" python -u run_part5.py \
    --dataset "$DATASET" \
    --checkpoint "$CHECKPOINT" \
    --stage compare \
    "${EXTRA_ARGS[@]}"
fi

echo "=== [$DATASET] requested Part 5 stages complete ==="
