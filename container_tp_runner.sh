#!/usr/bin/env bash
set -euo pipefail

# Loop over tensor parallel sizes and reuse container_perf_runner.sh for each run.

TP_LIST=(${TP_LIST:-4 8})
LOG_DIR=${LOG_DIR:-"perf_logs"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ARG="${1:-}"
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.855}
CONTAINER_PREFIX=${CONTAINER_PREFIX:-bench}

for tp in "${TP_LIST[@]}"; do
  echo "===> Running TP=${tp}"
  if ! TP="${tp}" MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" CONTAINER_NAME="${CONTAINER_PREFIX}-tp-${tp}" LOG_DIR="${LOG_DIR}" "${SCRIPT_DIR}/container_perf_runner.sh" ${MODEL_ARG:+${MODEL_ARG}}; then
    echo "[warn] TP=${tp} run failed; continuing to next TP."
    continue
  fi
done
