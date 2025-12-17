#!/usr/bin/env bash
set -euo pipefail

# Minimal container benchmark referencing perf_runner.sh; spins up a single
# sglang server container, waits for port 30000 to listen, hits the chat API,
# and records simple latency/tokens-per-second metrics before cleaning up.

IMAGE=${IMAGE:-"lmsysorg/sglang:latest"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
TP=${TP:-1}
CONTAINER_NAME=${CONTAINER_NAME:-"bench"}
PORT=${PORT:-30000}
PROMPT=${PROMPT:-"仿照过秦论写篇过隋论。请经过认真思考后写出，长度约为一万字。"}
MAX_TOKENS=${MAX_TOKENS:-10000}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-1}
WARMUP_MAX_TOKENS=${WARMUP_MAX_TOKENS:-256}
REPEAT_REQUESTS=${REPEAT_REQUESTS:-10}
TEMPERATURE=${TEMPERATURE:-0.7}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.855}
HF_CACHE=${HF_CACHE:-"$HOME/.cache/huggingface"}
WAIT_SECONDS=${WAIT_SECONDS:-7200}
POLL_INTERVAL=${POLL_INTERVAL:-5}
LOG_DIR=${LOG_DIR:-"perf_logs"}
BATCH_ENABLED=${BATCH_ENABLED:-1}
BATCH_PROMPTS=${BATCH_PROMPTS:-$'What is the meaning of life?\nWrite a one-sentence summary of why the sky is blue.\nExplain Docker port conflicts and how to fix “port is already allocated”.\nGive three tips to speed up Python code without changing results.\nDraft a polite email asking for an invoice correction.\nList five creative uses for a Raspberry Pi at home.\nExplain the difference between TCP and UDP in simple terms.\nWrite a haiku about winter in San Francisco.\nSuggest a 2-day itinerary for Vancouver in December.\nProvide a bash one-liner to find processes listening on port 30000.\nExplain what “post-money valuation” means with a numeric example.\nGenerate a JSON object describing a user profile (name, hobbies, skills).\nWrite a short bedtime story featuring a teddy bear astronaut.\nSummarize the key idea of “grokking” in neural networks in two sentences.\nGive a checklist for reviewing a startup equity agreement.\nExplain what “anti-dilution” is and the common types.\nPropose five A/B test ideas for a landing page conversion boost.\nWrite a Python function to compute moving average over a list.\nExplain how to batch requests with curl using JSON arrays.\nGive a concise definition of “representation theory” for ML engineers.\nSuggest a healthy 15-minute lunch recipe with common ingredients.'}
BATCH_MAX_TOKENS=${BATCH_MAX_TOKENS:-10000}
BATCH_TEMPERATURE=${BATCH_TEMPERATURE:-0.7}
BATCH_MODEL=${BATCH_MODEL:-"default"}
RUN_MODE=${RUN_MODE:-""}  # docker (default when available) or sbatch fallback
SERVICE_HOST=${SERVICE_HOST:-""}
SBATCH_SCRIPT=${SBATCH_SCRIPT:-"../andromeda.sbatch"}
SBATCH_JOB_ID=""
export MEM_FRACTION_STATIC

mkdir -p "$LOG_DIR"
run_id=$(date +%Y%m%d-%H%M%S)
RUN_LOG="$LOG_DIR/container-run-${run_id}.jsonl"

# Optional positional arg to override MODEL_PATH
if [[ $# -ge 1 ]]; then
  MODEL_PATH="$1"
  shift
fi

cleanup() {
  if [[ "${RUN_MODE}" == "docker" ]]; then
    command -v docker >/dev/null 2>&1 && docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  elif [[ -n "${SBATCH_JOB_ID}" ]]; then
    if command -v scancel >/dev/null 2>&1; then
      scancel "${SBATCH_JOB_ID}" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT

log_json() {
  echo "$1" >> "$RUN_LOG"
}

detect_run_mode() {
  if [[ -z "${RUN_MODE}" ]]; then
    if command -v docker >/dev/null 2>&1; then
      RUN_MODE="docker"
    elif command -v sbatch >/dev/null 2>&1; then
      RUN_MODE="sbatch"
    else
      echo "Neither docker nor sbatch is available; cannot start sglang service." >&2
      exit 1
    fi
  fi

  case "${RUN_MODE}" in
    docker)
      if ! command -v docker >/dev/null 2>&1; then
        echo "RUN_MODE=docker but docker command not found." >&2
        exit 1
      fi
      SERVICE_HOST=${SERVICE_HOST:-"127.0.0.1"}
      ;;
    sbatch)
      if ! command -v sbatch >/dev/null 2>&1; then
        echo "RUN_MODE=sbatch but sbatch command not found." >&2
        exit 1
      fi
      ;;
    *)
      echo "Unknown RUN_MODE=${RUN_MODE}. Expected docker or sbatch." >&2
      exit 1
      ;;
  esac
}

start_container() {
  # Remove any stale container with the same name first.
  docker ps -a --format '{{.Names}}' | grep -Fx "${CONTAINER_NAME}" >/dev/null 2>&1 && \
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

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

sbatch_job_state() {
  [[ -n "${SBATCH_JOB_ID}" ]] || return 1
  local state=""
  state=$(squeue -j "${SBATCH_JOB_ID}" -h -o "%T" 2>/dev/null | head -n1)
  if [[ -z "${state}" || "${state}" == "(null)" || "${state}" == "n/a" ]]; then
    state=$(scontrol show job "${SBATCH_JOB_ID}" 2>/dev/null | \
      awk '{
        for (i=1;i<=NF;i++){
          if ($i ~ /^JobState=/){
            sub(/^JobState=/,"",$i);
            print $i;
            exit;
          }
        }
      }' | head -n1)
  fi
  # Normalize common long forms to short codes used in checks.
  case "${state}" in
    RUNNING) state="R" ;;
    COMPLETING) state="CG" ;;
    COMPLETED) state="CD" ;;
    PENDING) state="PD" ;;
    FAILED) state="F" ;;
    CANCELLED) state="CA" ;;
    TIMEOUT) state="TO" ;;
    NODE_FAIL) state="NF" ;;
  esac
  [[ -n "${state}" ]] || return 1
  echo "${state}"
}

sbatch_job_host() {
  [[ -n "${SBATCH_JOB_ID}" ]] || return 1
  local host="" first_host=""
  host=$(squeue -j "${SBATCH_JOB_ID}" -h -o "%N" 2>/dev/null | head -n1)
  if [[ -z "${host}" || "${host}" == "(null)" || "${host}" == "n/a" ]]; then
    host=$(scontrol show job "${SBATCH_JOB_ID}" 2>/dev/null | \
      grep -o 'NodeList=[^ ]*' | head -n1 | cut -d= -f2)
  fi
  if [[ -z "${host}" || "${host}" == "(null)" || "${host}" == "n/a" ]]; then
    host=$(scontrol show job "${SBATCH_JOB_ID}" -o 2>/dev/null | \
      sed -n 's/.*BatchHost=\\?\\([^ ]*\\).*/\\1/p' | head -n1)
  fi
  if [[ "${host}" == "None assigned" ]]; then
    host=""
  fi
  if [[ -z "${host}" ]]; then
    return 1
  fi
  first_host=$(echo "${host}" | awk -F',' '{print $1}')
  if [[ "${first_host}" == *"["*"]"* ]]; then
    first_host=$(scontrol show hostnames "${first_host}" 2>/dev/null | head -n1)
  fi
  [[ -n "${first_host}" ]] || return 1
  echo "${first_host}"
}

resolve_host_ip() {
  local host="$1"
  [[ -n "${host}" ]] || return 1
  getent hosts "${host}" 2>/dev/null | awk 'NR==1 {print $1}'
}

wait_for_sbatch_start() {
  local elapsed=0 state node
  while (( elapsed < WAIT_SECONDS )); do
    state=$(sbatch_job_state || true)
    if [[ -z "${state}" ]]; then
      echo "sbatch job ${SBATCH_JOB_ID} is no longer in queue." >&2
      return 1
    fi
    if [[ "${state}" == "R" || "${state}" == "CG" ]]; then
      node=$(sbatch_job_host || true)
      if [[ -z "${SERVICE_HOST}" && -n "${node}" && "${node}" != "(null)" && "${node}" != "n/a" ]]; then
        SERVICE_HOST="${node}"
      fi
      host_to_print="${SERVICE_HOST:-${node}}"
      if [[ -n "${host_to_print}" ]]; then
        ip=$(resolve_host_ip "${host_to_print}" || true)
        echo "sbatch job ${SBATCH_JOB_ID} running (state=${state}) on host ${host_to_print} ${ip:+(ip ${ip})}."
      else
        echo "sbatch job ${SBATCH_JOB_ID} running (state=${state}) on host <unknown>."
      fi
      return 0
    fi
    if [[ "${state}" =~ ^(F|TO|NF|CA|CD)$ ]]; then
      echo "sbatch job ${SBATCH_JOB_ID} entered terminal state (${state})." >&2
      return 1
    fi
    sleep "${POLL_INTERVAL}"
    elapsed=$((elapsed + POLL_INTERVAL))
  done
  echo "Timed out waiting for sbatch job ${SBATCH_JOB_ID} to start after ${WAIT_SECONDS}s." >&2
  return 1
}

start_sbatch_service() {
  if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
    echo "sbatch script ${SBATCH_SCRIPT} not found." >&2
    exit 1
  fi
  local submit_out
  submit_out=$(MODEL_PATH="${MODEL_PATH}" PORT="${PORT}" MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" IMAGE="${IMAGE}" TP="${TP}" HF_CACHE="${HF_CACHE}" sbatch --parsable "${SBATCH_SCRIPT}")
  SBATCH_JOB_ID=$(echo "${submit_out}" | tail -n1 | tr -d '[:space:]')
  if [[ -z "${SBATCH_JOB_ID}" ]]; then
    echo "Failed to parse sbatch job id from output: ${submit_out}" >&2
    exit 1
  fi
  echo "Submitted sbatch job ${SBATCH_JOB_ID} via ${SBATCH_SCRIPT}."
  wait_for_sbatch_start
  if [[ -n "${SERVICE_HOST}" ]]; then
    local ip=""
    ip=$(resolve_host_ip "${SERVICE_HOST}" || true)
    echo "sbatch job ${SBATCH_JOB_ID} host: ${SERVICE_HOST} ${ip:+(ip ${ip})}"
  fi
}

wait_for_ready() {
  local elapsed=0
  local msg=""
  local host="${SERVICE_HOST:-127.0.0.1}"
  local state=""
  local last_host_noted=""
  local ip=""
  while (( elapsed < WAIT_SECONDS )); do
    msg="Waiting for port ${PORT} on ${host}${ip:+ (ip ${ip})}... waited ${elapsed}s"
    printf "\r%s" "${msg}"
    if [[ "${RUN_MODE}" == "docker" ]]; then
      if ! container_running; then
        printf "\n"
        echo "[${CONTAINER_NAME}] not running; logs below:"
        docker logs "${CONTAINER_NAME}" || true
        return 1
      fi
      # Prefer in-container port check; fallback to host curl if netstat/ss missing.
      if docker exec "${CONTAINER_NAME}" sh -c "ss -lntp 2>/dev/null || netstat -lntp 2>/dev/null" | grep -q ":${PORT}"; then
        printf "\rWaiting for port ${PORT}... ready after ${elapsed}s\n"
        return 0
      fi
    else
      state=$(sbatch_job_state || true)
      if [[ -z "${state}" ]]; then
        printf "\n"
        echo "sbatch job ${SBATCH_JOB_ID} disappeared from queue while waiting." >&2
        return 1
      fi
      if [[ "${state}" =~ ^(F|TO|NF|CA|CD)$ ]]; then
        printf "\n"
        echo "sbatch job ${SBATCH_JOB_ID} entered terminal state (${state}) while waiting." >&2
        return 1
      fi
      if [[ -z "${host}" ]]; then
        host=$(sbatch_job_host || true)
      fi
      if [[ -n "${host}" && "${SERVICE_HOST}" != "${host}" ]]; then
        SERVICE_HOST="${host}"
        if [[ "${last_host_noted}" != "${host}" ]]; then
          ip=$(resolve_host_ip "${host}" || true)
          last_host_noted="${host}"
        fi
      fi
    fi
    local target="${ip:-${host}}"
    if [[ -n "${target}" ]] && curl -sf "http://${target}:${PORT}/v1/models" >/dev/null 2>&1; then
      printf "\rWaiting for port ${PORT} on ${host}${ip:+ (ip ${ip})}... ready after ${elapsed}s\n"
      return 0
    fi
    sleep "${POLL_INTERVAL}"
    elapsed=$((elapsed + POLL_INTERVAL))
  done
  printf "\n"
  echo "[${CONTAINER_NAME}] port ${PORT} not ready after ${WAIT_SECONDS}s"
  if [[ "${RUN_MODE}" == "docker" ]]; then
    docker logs "${CONTAINER_NAME}" || true
  elif [[ -n "${SBATCH_JOB_ID}" ]]; then
    scontrol show job "${SBATCH_JOB_ID}" || true
  fi
  return 1
}

build_payload() {
  local payload_file prompt max_tokens
  prompt="${1:-$PROMPT}"
  max_tokens="${2:-$MAX_TOKENS}"
  payload_file=$(mktemp)
  MODEL_PATH="${MODEL_PATH}" PROMPT="${prompt}" MAX_TOKENS="${max_tokens}" TEMPERATURE="${TEMPERATURE}" PAYLOAD_FILE="${payload_file}" python3 - <<'PY'
import json, os, pathlib
payload = {
    "model": os.environ["MODEL_PATH"],
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": float(os.environ["TEMPERATURE"]),
}
pathlib.Path(os.environ["PAYLOAD_FILE"]).write_text(json.dumps(payload))
PY
  echo "${payload_file}"
}

run_request() {
  local prompt="${1:-$PROMPT}"
  local max_tokens="${2:-$MAX_TOKENS}"
  local iter_label="${3:-""}"
  local host="${SERVICE_HOST:-127.0.0.1}"
  local payload_file resp_file t_start t_end metrics
  payload_file=$(build_payload "${prompt}" "${max_tokens}")
  resp_file=$(mktemp)
  t_start=$(python3 - <<'PY'
import time; print(time.time())
PY
)
  curl -s -X POST "http://${host}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data @"${payload_file}" > "${resp_file}"
  t_end=$(python3 - <<'PY'
import time; print(time.time())
PY
)
  metrics=$(RESP_FILE="${resp_file}" MODEL="${MODEL_PATH}" TP="${TP}" MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" T_START="${t_start}" T_END="${t_end}" ITER_LABEL="${iter_label}" python3 - <<'PY'
import json, os, pathlib
resp = pathlib.Path(os.environ["RESP_FILE"]).read_text()
t_start = float(os.environ["T_START"])
t_end = float(os.environ["T_END"])
elapsed = max(t_end - t_start, 0.0)
out = {
    "model": os.environ["MODEL"],
    "tp": int(os.environ["TP"]),
    "mem_fraction_static": float(os.environ.get("MEM_FRACTION_STATIC", 0.0)),
    "iteration": os.environ.get("ITER_LABEL") or None,
    "status": "ok",
    "elapsed_sec": elapsed,
    "tokens_per_sec": None,
    "completion_tokens": None,
    "prompt_tokens": None,
    "total_tokens": None,
    "response_head": resp[:400],
}
try:
    data = json.loads(resp)
    usage = data.get("usage", {})
    ct = usage.get("completion_tokens")
    out["completion_tokens"] = ct
    out["prompt_tokens"] = usage.get("prompt_tokens")
    out["total_tokens"] = usage.get("total_tokens")
    if ct is not None and elapsed > 0:
        out["tokens_per_sec"] = ct / elapsed
except Exception as e:
    out["status"] = "error"
    out["error"] = str(e)
print(json.dumps(out))
PY
)
  rm -f "${payload_file}" "${resp_file}"
  echo "${metrics}"
}

build_batch_payload() {
  local payload_file
  payload_file=$(mktemp)
  BATCH_MODEL="${BATCH_MODEL}" BATCH_PROMPTS="${BATCH_PROMPTS}" BATCH_MAX_TOKENS="${BATCH_MAX_TOKENS}" BATCH_TEMPERATURE="${BATCH_TEMPERATURE}" PAYLOAD_FILE="${payload_file}" python3 - <<'PY'
import json, os, pathlib
prompts = [p for p in os.environ["BATCH_PROMPTS"].splitlines() if p.strip()]
payload = {
    "model": os.environ["BATCH_MODEL"],
    "prompt": prompts,
    "max_tokens": int(os.environ["BATCH_MAX_TOKENS"]),
    "temperature": float(os.environ["BATCH_TEMPERATURE"]),
}
pathlib.Path(os.environ["PAYLOAD_FILE"]).write_text(json.dumps(payload))
PY
  echo "${payload_file}"
}

run_batch_request() {
  local iter_label="${1:-"batch-1"}"
  local host="${SERVICE_HOST:-127.0.0.1}"
  local payload_file resp_file t_start t_end metrics
  payload_file=$(build_batch_payload)
  resp_file=$(mktemp)
  t_start=$(python3 - <<'PY'
import time; print(time.time())
PY
)
  curl -s -X POST "http://${host}:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer EMPTY" \
    --data @"${payload_file}" > "${resp_file}"
  t_end=$(python3 - <<'PY'
import time; print(time.time())
PY
)
  metrics=$(RESP_FILE="${resp_file}" MODEL="${MODEL_PATH}" TP="${TP}" MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" ITER_LABEL="${iter_label}" T_START="${t_start}" T_END="${t_end}" BATCH_PROMPTS="${BATCH_PROMPTS}" python3 - <<'PY'
import json, os, pathlib
resp = pathlib.Path(os.environ["RESP_FILE"]).read_text()
t_start = float(os.environ["T_START"])
t_end = float(os.environ["T_END"])
elapsed = max(t_end - t_start, 0.0)
prompts = [p for p in os.environ["BATCH_PROMPTS"].splitlines() if p.strip()]
out = {
    "model": os.environ["MODEL"],
    "tp": int(os.environ["TP"]),
    "mem_fraction_static": float(os.environ.get("MEM_FRACTION_STATIC", 0.0)),
    "iteration": os.environ.get("ITER_LABEL") or None,
    "batch_size": len(prompts),
    "status": "ok",
    "elapsed_sec": elapsed,
    "tokens_per_sec": None,
    "tokens_per_sec_per_prompt": None,
    "completion_tokens": None,
    "prompt_tokens": None,
    "total_tokens": None,
    "response_head": resp[:400],
}
try:
    data = json.loads(resp)
    usage = data.get("usage", {})
    ct = usage.get("completion_tokens")
    out["completion_tokens"] = ct
    out["prompt_tokens"] = usage.get("prompt_tokens")
    out["total_tokens"] = usage.get("total_tokens")
    choices = data.get("choices", [])
    if choices:
        out["choices_text"] = [c.get("text", "")[:200] for c in choices]
    if ct is not None and elapsed > 0:
        out["tokens_per_sec"] = ct / elapsed
        if out["batch_size"] > 0:
            out["tokens_per_sec_per_prompt"] = (ct / out["batch_size"]) / elapsed
except Exception as e:
    out["status"] = "error"
    out["error"] = str(e)
print(json.dumps(out))
PY
)
  rm -f "${payload_file}" "${resp_file}"
  echo "${metrics}"
}

detect_run_mode

startup_start=$(python3 - <<'PY'
import time; print(time.time())
PY
)

if [[ "${RUN_MODE}" == "docker" ]]; then
  echo "Starting docker container ${CONTAINER_NAME} with model ${MODEL_PATH} tp=${TP}..."
  start_container
  echo "Waiting for port ${PORT} inside container (docker exec netstat)..."
else
  echo "Docker not available; submitting sbatch job via ${SBATCH_SCRIPT} for model ${MODEL_PATH} tp=${TP}..."
  start_sbatch_service
  host_label="${SERVICE_HOST:-<pending>}"
  echo "Waiting for port ${PORT} on host ${host_label} exposed by sbatch job ${SBATCH_JOB_ID}..."
fi

if ! wait_for_ready; then
  log_json "{\"model\":\"${MODEL_PATH}\",\"tp\":${TP},\"mem_fraction_static\":${MEM_FRACTION_STATIC},\"status\":\"failed_start\"}"
  exit 1
fi

startup_end=$(python3 - <<'PY'
import time; print(time.time())
PY
)
STARTUP_WAIT_SEC=$(python3 - <<'PY'
import os
start=float(os.environ["START_TIME"])
end=float(os.environ["END_TIME"])
print(max(end - start, 0.0))
PY
) START_TIME="${startup_start}" END_TIME="${startup_end}"
export STARTUP_WAIT_SEC
echo "Startup wait time: ${STARTUP_WAIT_SEC}s"

if [[ "${RUN_MODE}" == "docker" ]]; then
  echo "Container status: $(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")"
else
  echo "sbatch job ${SBATCH_JOB_ID} is running; using host ${SERVICE_HOST:-unknown}."
fi

# Warmup requests (shorter outputs)
for ((i=1; i<=WARMUP_REQUESTS; i++)); do
  echo "Warmup ${i}/${WARMUP_REQUESTS} (max_tokens=${WARMUP_MAX_TOKENS})..."
  wres=$(run_request "${PROMPT}" "${WARMUP_MAX_TOKENS}" "warmup-${i}")
  log_json "${wres}"
done

# Main repeated runs
tokens_list=()
elapsed_list=()
for ((i=1; i<=REPEAT_REQUESTS; i++)); do
  echo "Benchmark run ${i}/${REPEAT_REQUESTS} (max_tokens=${MAX_TOKENS})..."
  res=$(run_request "${PROMPT}" "${MAX_TOKENS}" "run-${i}")
  log_json "${res}"
  echo "Result: ${res}"
  tps=$(METRICS="${res}" python3 - <<'PY'
import json, os
try:
    data = json.loads(os.environ["METRICS"])
    v = data.get("tokens_per_sec")
    if v is None:
        raise SystemExit("")
    print(v)
except Exception:
    pass
PY
)
  elapsed=$(METRICS="${res}" python3 - <<'PY'
import json, os
try:
    data = json.loads(os.environ["METRICS"])
    v = data.get("elapsed_sec")
    if v is None:
        raise SystemExit("")
    print(v)
except Exception:
    pass
PY
)
  [[ -n "${tps}" ]] && tokens_list+=("${tps}")
  [[ -n "${elapsed}" ]] && elapsed_list+=("${elapsed}")
done

# Batched completions test (optional)
if [[ "${BATCH_ENABLED}" -eq 1 ]]; then
  batch_size=$(BATCH_PROMPTS="${BATCH_PROMPTS}" python3 - <<'PY'
import os
prompts = [p for p in os.environ["BATCH_PROMPTS"].splitlines() if p.strip()]
print(len(prompts))
PY
)
  echo "Running batched completion (${BATCH_MAX_TOKENS} tokens, port ${PORT}, batch size ${batch_size})..."
  bres=$(run_batch_request "batch-1")
  log_json "${bres}"
  echo "Batch result: ${bres}"
fi

# Summary stats
TPS_LIST="${tokens_list[*]}"
ELAPSED_LIST="${elapsed_list[*]}"
summary=$(TPS_LIST="${TPS_LIST}" ELAPSED_LIST="${ELAPSED_LIST}" MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" STARTUP_WAIT_SEC="${STARTUP_WAIT_SEC}" python3 - <<'PY'
import math, os, json
def stats(vals):
    if not vals:
        return {"count": 0, "mean": None, "std": None}
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return {"count": len(vals), "mean": mean, "std": std}

tps_vals = [float(x) for x in os.environ.get("TPS_LIST", "").split() if x]
elapsed_vals = [float(x) for x in os.environ.get("ELAPSED_LIST", "").split() if x]
out = {
    "mem_fraction_static": float(os.environ.get("MEM_FRACTION_STATIC", 0.0)),
    "tokens_per_sec_stats": stats(tps_vals),
    "elapsed_sec_stats": stats(elapsed_vals),
    "startup_wait_sec": float(os.environ.get("STARTUP_WAIT_SEC", 0.0)),
}
print(json.dumps(out))
PY
)
log_json "${summary}"
echo "Summary stats: ${summary}"

if [[ "${RUN_MODE}" == "docker" ]]; then
  echo "Stopping container ${CONTAINER_NAME}..."
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
elif [[ -n "${SBATCH_JOB_ID}" ]]; then
  echo "Cancelling sbatch job ${SBATCH_JOB_ID}..."
  scancel "${SBATCH_JOB_ID}" >/dev/null 2>&1 || true
fi
