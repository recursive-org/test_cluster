#!/usr/bin/env bash
# Start 8 Tongyi containers, one per GPU, ports starting from BASE_PORT.
# Calls start_tongyi.sh for each container.

set -euo pipefail

IMAGE=${IMAGE:-"lmsysorg/sglang:latest"}
MODEL_PATH=${MODEL_PATH:-"Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"}
TP=${TP:-1}
CONTAINER_PREFIX=${CONTAINER_PREFIX:-"tongyi"}
BASE_PORT=${BASE_PORT:-30000}
NUM_GPUS=${NUM_GPUS:-8}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.855}
HF_CACHE=${HF_CACHE:-"$HOME/.cache/huggingface"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_ONE_SCRIPT="${SCRIPT_DIR}/start_tongyi.sh"

# Optional positional arg to override MODEL_PATH
if [[ $# -ge 1 ]]; then
  MODEL_PATH="$1"
fi

if [[ ! -x "${START_ONE_SCRIPT}" ]]; then
  echo "Missing start script: ${START_ONE_SCRIPT}"
  exit 1
fi

echo "Starting ${NUM_GPUS} containers for model=${MODEL_PATH} with base port=${BASE_PORT}..."
for ((i = 0; i < NUM_GPUS; i++)); do
  port=$((BASE_PORT + i))
  name="${CONTAINER_PREFIX}-${i}"
  IMAGE="${IMAGE}" \
  MODEL_PATH="${MODEL_PATH}" \
  TP="${TP}" \
  CONTAINER_NAME="${name}" \
  PORT="${port}" \
  GPU_DEVICE="device=${i}" \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
  HF_CACHE="${HF_CACHE}" \
  "${START_ONE_SCRIPT}" &
  echo "Started ${name} on GPU ${i}, port ${port}"
done

wait
echo "All containers started."
