#!/usr/bin/env bash
# Hard-coded runs derived from perf_summary.html rows.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=${LOG_DIR:-"perf_logs"}
PORT_BASE=${PORT_BASE:-30000}

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
for cfg in "${configs[@]}"; do
  IFS=":" read -r model mem tp <<<"${cfg}"
  port=$((PORT_BASE + idx))
  cname_prefix="bench-${idx}"
  echo "===> Running model=${model} tp=${tp} mem_fraction_static=${mem:-"(default)"} port=${port}"
  MEM_FRACTION_STATIC="${mem}" TP_LIST="${tp}" LOG_DIR="${LOG_DIR}" PORT="${port}" CONTAINER_PREFIX="${cname_prefix}" "${SCRIPT_DIR}/container_tp_runner.sh" "${model}" || {
    echo "[warn] run failed for model=${model} tp=${tp}"
  }
  idx=$((idx + 1))
done
