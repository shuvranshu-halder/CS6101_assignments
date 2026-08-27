#!/bin/bash
# Usage: 
#   bash scripts/run_1to4.sh scifact
#   bash scripts/run_1to4.sh scifact --stage part4a

set -e

DATASET=$1
shift || true

if [ -z "$DATASET" ]; then
  echo "Usage: bash scripts/run_1to4.sh <scifact|fever|hotpotqa> [extra python args]"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="2,3"
cd "$(dirname "$0")/.."

# Check if a custom --stage flag was passed in extra args
HAS_STAGE=0
for arg in "$@"; do
  if [ "$arg" == "--stage" ]; then
    HAS_STAGE=1
    break
  fi
done

if [ "$HAS_STAGE" -eq 1 ]; then
  echo "=== [$DATASET] Executing specified stage override with args: $@ ==="
  python run_parts1to4.py --dataset "$DATASET" "$@"
else
  echo "=== [$DATASET] Parts 1-3 + 4a ==="
  python run_parts1to4.py --dataset "$DATASET" --stage part1 part2 part3 part4a "$@"

  echo "=== [$DATASET] Part 4b (HyDE generation + retrieval) ==="
  python run_parts1to4.py --dataset "$DATASET" --stage part4b_generate part4b_run \
    --hyde_samples 4 --hyde_model "Qwen/Qwen2.5-7B-Instruct" "$@"

  echo "=== [$DATASET] Parts 1-4 complete. Results: results/results_${DATASET}.md ==="
fi