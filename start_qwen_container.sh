#!/usr/bin/env bash
# Spin up a Qwen/Qwen3-8B sglang server in Docker, wait for readiness, then exit.
# Reuses container bootstrap logic from container_perf_runner.sh without benchmarking.

set -euo pipefail

IMAGE=${IMAGE:-"lmsysorg/sglang:latest"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
TP=${TP:-1}
CONTAINER_NAME=${CONTAINER_NAME:-"qwen-server"}
PORT=${PORT:-30000}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.855}
HF_CACHE=${HF_CACHE:-"$HOME/.cache/huggingface"}
WAIT_SECONDS=${WAIT_SECONDS:-600}
POLL_INTERVAL=${POLL_INTERVAL:-5}

# Optional positional arg to override MODEL_PATH
if [[ $# -ge 1 ]]; then
  MODEL_PATH="$1"
fi

remove_stale() {
  docker ps -a --format '{{.Names}}' | grep -Fx "${CONTAINER_NAME}" >/dev/null 2>&1 || return 0
  echo "Removing existing container ${CONTAINER_NAME}..."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
}

start_container() {
  docker run --gpus all --shm-size 32g -it --rm -p "${PORT}:30000" -d \
    --name "${CONTAINER_NAME}" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    "${IMAGE}" \
    python3 -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --host 0.0.0.0 --port 30000 \
      --tp "${TP}" \
      --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      --trust-remote-code
}

container_running() {
  docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null | grep -q true
}

wait_for_ready() {
  local elapsed=0
  while (( elapsed < WAIT_SECONDS )); do
    if ! container_running; then
      echo
      echo "[${CONTAINER_NAME}] exited early. Logs:"
      docker logs "${CONTAINER_NAME}" || true
      return 1
    fi
    # Check inside container first
    if docker exec "${CONTAINER_NAME}" sh -c "ss -lntp 2>/dev/null || netstat -lntp 2>/dev/null" | grep -q ":${PORT}"; then
      echo "Ready: container listening on ${PORT} after ${elapsed}s"
      return 0
    fi
    # Fallback host probe
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "Ready: host probe succeeded on ${PORT} after ${elapsed}s"
      return 0
    fi
    printf "\rWaiting for port %s... (%ss elapsed)" "${PORT}" "${elapsed}"
    sleep "${POLL_INTERVAL}"
    elapsed=$((elapsed + POLL_INTERVAL))
  done
  echo
  echo "[${CONTAINER_NAME}] not ready after ${WAIT_SECONDS}s"
  docker logs "${CONTAINER_NAME}" || true
  return 1
}

echo "Starting ${CONTAINER_NAME} with model=${MODEL_PATH} tp=${TP} on port ${PORT}..."
remove_stale
start_container
wait_for_ready

echo "Container ${CONTAINER_NAME} is up. You can connect at http://127.0.0.1:${PORT}/v1"
