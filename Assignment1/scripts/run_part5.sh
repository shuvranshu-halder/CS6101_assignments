#!/bin/bash
# Runs Part 5 (SPLADE) for one dataset, sharding corpus encoding across
# multiple GPUs in parallel (this is the stage that actually needs multi-GPU
# for the big corpora — FEVER/HotpotQA are ~5M docs, MSMARCO 8.8M for extra credit).
#
# Usage: bash scripts/run_part5.sh <dataset>
#   bash scripts/run_part5.sh scifact
#   bash scripts/run_part5.sh fever
set -e

DATASET=$1
if [ -z "$DATASET" ]; then
  echo "Usage: bash scripts/run_part5.sh <scifact|fever|hotpotqa>"
  exit 1
fi

# ---------------------------------------------------------------------------
# GPU CONFIG — edit per dataset. scifact is tiny (5K docs, 1 GPU is plenty);
# fever/hotpotqa (~5M docs) benefit from sharding across all available GPUs.
# For extra-credit MSMARCO (8.8M docs), bump GPU_IDS to all 8 GPUs.
# ---------------------------------------------------------------------------
case "$DATASET" in
  scifact)
    GPU_IDS=(5)
    BATCH_SIZE=128
    ;;
  fever|hotpotqa)
    GPU_IDS=(0 1 2 3)
    BATCH_SIZE=64
    ;;
  msmarco)  # extra credit
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

echo "=== [$DATASET] encoding corpus across ${NUM_SHARDS} GPU(s): ${GPU_IDS[*]} ==="
PIDS=()
for shard_id in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$shard_id]}"
  echo "  shard $shard_id -> GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u run_part5.py \
    --dataset "$DATASET" --checkpoint "$CHECKPOINT" \
    --stage encode --shard_id "$shard_id" --num_shards "$NUM_SHARDS" \
    --batch_size "$BATCH_SIZE" \
    > logs/"$DATASET"_splade_shard"$shard_id".log 2>&1 &
  PIDS+=($!)
done

echo "=== waiting for all shards to finish encoding ==="
for pid in "${PIDS[@]}"; do wait "$pid"; done

echo "=== [$DATASET] retrieval + expansion-term comparison (single GPU 0) ==="
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" python -u run_part5.py \
  --dataset "$DATASET" --checkpoint "$CHECKPOINT" --stage retrieve compare

echo "=== [$DATASET] Part 5 complete. Results appended to results/results_${DATASET}.md ==="
