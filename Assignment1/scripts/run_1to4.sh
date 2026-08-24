#!/bin/bash
# Runs Parts 1-4 for one dataset.
# Usage: bash scripts/run_1to4.sh <dataset> [extra python args...]
#   bash scripts/run_1to4.sh scifact
#   bash scripts/run_1to4.sh fever --stage part2 part3
set -e

DATASET=$1
shift || true

if [ -z "$DATASET" ]; then
  echo "Usage: bash scripts/run_1to4.sh <scifact|fever|hotpotqa> [extra args]"
  exit 1
fi

# ---------------------------------------------------------------------------
# GPU CONFIG — Parts 1-3 and 4a are CPU/Lucene only. Only Part 4b's local
# HyDE model needs a GPU. Edit these to change GPU allocation / server location.
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="2"          # which GPU(s) the vLLM server uses
# VLLM_TENSOR_PARALLEL_SIZE=1              # set >1 if splitting the 7B model across GPUs
# VLLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
# VLLM_PORT=8000

# If the vLLM server runs on THIS same machine (default, most common case),
# leave as-is. If it runs elsewhere — a different node in your cluster, a
# separate pod, etc. — set VLLM_HOST to that machine's hostname/IP, and set
# SKIP_VLLM_SERVER=1 so this script doesn't try to launch a duplicate server
# locally, e.g.:
#   VLLM_HOST=gpu-node-03 SKIP_VLLM_SERVER=1 bash scripts/run_1to4.sh scifact
# VLLM_HOST="${VLLM_HOST:-localhost}"
# SKIP_VLLM_SERVER="${SKIP_VLLM_SERVER:-0}"
# VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

cd "$(dirname "$0")/.."

echo "=== [$DATASET] Parts 1-3 + 4a (no GPU needed) ==="
python run_parts1to4.py --dataset "$DATASET" --stage part1 part2 part3 part4a "$@"

echo "=== [$DATASET] Part 4b (HyDE generation + retrieval, direct transformers, no server) ==="
python run_parts1to4.py --dataset "$DATASET" --stage part4b_generate part4b_run \
  --hyde_samples 4 --hyde_model "Qwen/Qwen2.5-7B-Instruct" "$@"

# echo "waiting for vLLM server to come up at $VLLM_BASE_URL ..."
# until curl -s "${VLLM_BASE_URL}/models" > /dev/null; do sleep 5; done

echo "=== [$DATASET] Part 4b (HyDE generation + retrieval) ==="
# python run_parts1to4.py --dataset "$DATASET" --stage part4b_generate part4b_run \
#   --hyde_samples 4 --vllm_base_url "$VLLM_BASE_URL" "$@"

# if [ -n "$VLLM_PID" ]; then
#   echo "=== [$DATASET] stopping vLLM server ==="
#   kill $VLLM_PID
# fi

echo "=== [$DATASET] Parts 1-4 complete. Results: results/results_${DATASET}.md ==="
