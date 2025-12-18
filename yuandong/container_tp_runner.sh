#!/usr/bin/env bash
set -euo pipefail

# Loop over tensor parallel sizes and reuse container_perf_runner.sh for each run.

TP_LIST=(${TP_LIST:-4 8})
LOG_DIR=${LOG_DIR:-"perf_logs"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ARG="${1:-}"
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.855}
CONTAINER_PREFIX=${CONTAINER_PREFIX:-bench}
PARALLEL=${PARALLEL:-0}
PORT_BASE=${PORT_BASE:-30000}
PORT_STEP=${PORT_STEP:-100}

pids=()
labels=()
idx=0

for tp in "${TP_LIST[@]}"; do
  port=$((PORT_BASE + idx * PORT_STEP))
  run_name="${CONTAINER_PREFIX}-tp-${tp}"
  echo "===> Running TP=${tp} (port ${port})${PARALLEL:+ [parallel]}"
  if [[ "${PARALLEL}" -eq 1 ]]; then
    (
      TP="${tp}" PORT="${port}" MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" CONTAINER_NAME="${run_name}" LOG_DIR="${LOG_DIR}" "${SCRIPT_DIR}/container_perf_runner.sh" ${MODEL_ARG:+${MODEL_ARG}}
    ) &
    pids+=("$!")
    labels+=("TP=${tp}")
  else
    if ! TP="${tp}" PORT="${port}" MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" CONTAINER_NAME="${run_name}" LOG_DIR="${LOG_DIR}" "${SCRIPT_DIR}/container_perf_runner.sh" ${MODEL_ARG:+${MODEL_ARG}}; then
      echo "[warn] TP=${tp} run failed; continuing to next TP."
    fi
  fi
  idx=$((idx + 1))
done

if [[ "${PARALLEL}" -eq 1 ]]; then
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    label="${labels[$i]}"
    if wait "${pid}"; then
      echo "[ok] ${label} finished."
    else
      echo "[warn] ${label} failed."
    fi
  done
fi
