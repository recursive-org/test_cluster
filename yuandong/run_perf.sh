#!/usr/bin/env bash
# Hard-coded runs derived from perf_summary.html rows.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=${LOG_DIR:-"perf_logs"}
PORT_BASE=${PORT_BASE:-30000}
PORT_STEP=${PORT_STEP:-10}
PARALLEL=${PARALLEL:-0}
STAGED=${STAGED:-0} # if 1: start all services, wait for all ready, then benchmark
STAGED_MAX_LAUNCH=${STAGED_MAX_LAUNCH:-2} # max concurrent startups when STAGED=1

# model mem_fraction_static tp
configs=(
  "Qwen/Qwen3-8B::1"
  "Qwen/Qwen3-8B::2"
  "Qwen/Qwen3-8B::4"
  "Qwen/Qwen3-8B::8"
  "deepseek-ai/DeepSeek-V3-0324::8"
  "deepseek-ai/DeepSeek-V3-0324:0.9:4"
  "deepseek-ai/DeepSeek-V3-0324:0.9:8"
  "deepseek-ai/DeepSeek-V3.2:0.9:8"
  "nvidia/DeepSeek-V3-0324-NVFP4::4"
  "nvidia/DeepSeek-V3-0324-NVFP4::8"
)

idx=0
pids=()
labels=()
state_files=()

if [[ "${STAGED}" -eq 1 ]]; then
  mkdir -p "${LOG_DIR}/state"
  echo "===> STAGED=1: launching all jobs first, then benchmarking."
  echo "     Limiting concurrent launches to ${STAGED_MAX_LAUNCH}"

  launch_failed=0
  active_pids=()
  active_labels=()

  wait_one_active() {
    local pid label
    pid="${active_pids[0]}"
    label="${active_labels[0]}"
    if wait "${pid}"; then
      echo "[ok] ${label} ready."
    else
      echo "[warn] ${label} failed."
      launch_failed=1
    fi
    active_pids=("${active_pids[@]:1}")
    active_labels=("${active_labels[@]:1}")
  }

  wait_for_slot() {
    while (( ${#active_pids[@]} >= STAGED_MAX_LAUNCH )); do
      wait_one_active
    done
  }

  for cfg in "${configs[@]}"; do
    IFS=":" read -r model mem tp <<<"${cfg}"
    port=$((PORT_BASE + idx * PORT_STEP))
    cname="bench-${idx}-tp-${tp}"
    state_file="${LOG_DIR}/state/${idx}-tp-${tp}.env"
    echo "===> Launching model=${model} tp=${tp} mem_fraction_static=${mem:-"(default)"} port=${port}"
    wait_for_slot
    (
      PHASE=start STATE_FILE="${state_file}" SKIP_CLEANUP=1 \
        MEM_FRACTION_STATIC="${mem}" TP="${tp}" LOG_DIR="${LOG_DIR}" PORT="${port}" CONTAINER_NAME="${cname}" \
        "${SCRIPT_DIR}/container_perf_runner.sh" "${model}"
    ) &
    active_pids+=("$!")
    active_labels+=("launch model=${model} tp=${tp}")
    state_files+=("${state_file}") # preserves order for benchmarking
    idx=$((idx + 1))
  done

  # Drain any remaining launches
  while (( ${#active_pids[@]} > 0 )); do
    wait_one_active
  done
  if [[ "${launch_failed}" -eq 1 ]]; then
    echo "[warn] One or more launches failed; benchmarking will skip missing state files." >&2
  fi

  for state_file in "${state_files[@]}"; do
    if [[ ! -f "${state_file}" ]]; then
      echo "[warn] Missing state file ${state_file}; skipping benchmark." >&2
      continue
    fi
    set -a
    # shellcheck disable=SC1090
    source "${state_file}"
    set +a

    echo "===> Benchmarking model=${MODEL_PATH} tp=${TP} on ${SERVICE_HOST:-127.0.0.1}:${PORT} (${RUN_MODE})"
    if ! PHASE=bench SKIP_CLEANUP=1 "${SCRIPT_DIR}/container_perf_runner.sh"; then
      echo "[warn] benchmark failed for model=${MODEL_PATH} tp=${TP}"
    fi

    echo "===> Cleaning up model=${MODEL_PATH} tp=${TP} (${RUN_MODE})"
    PHASE=cleanup "${SCRIPT_DIR}/container_perf_runner.sh" || true
  done

  exit 0
fi

for cfg in "${configs[@]}"; do
  IFS=":" read -r model mem tp <<<"${cfg}"
  port=$((PORT_BASE + idx * PORT_STEP))
  cname_prefix="bench-${idx}"
  echo "===> Running model=${model} tp=${tp} mem_fraction_static=${mem:-"(default)"} port=${port}${PARALLEL:+ [parallel]}"
  if [[ "${PARALLEL}" -eq 1 ]]; then
    (
      MEM_FRACTION_STATIC="${mem}" TP_LIST="${tp}" LOG_DIR="${LOG_DIR}" PORT_BASE="${port}" PORT_STEP="${PORT_STEP}" CONTAINER_PREFIX="${cname_prefix}" PARALLEL=1 "${SCRIPT_DIR}/container_tp_runner.sh" "${model}"
    ) &
    pids+=("$!")
    labels+=("model=${model} tp=${tp}")
  else
    MEM_FRACTION_STATIC="${mem}" TP_LIST="${tp}" LOG_DIR="${LOG_DIR}" PORT_BASE="${port}" PORT_STEP="${PORT_STEP}" CONTAINER_PREFIX="${cname_prefix}" PARALLEL=0 "${SCRIPT_DIR}/container_tp_runner.sh" "${model}" || {
      echo "[warn] run failed for model=${model} tp=${tp}"
    }
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
