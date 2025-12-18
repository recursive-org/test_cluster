#!/bin/bash
#SBATCH --job-name=pstate-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=02:00:00
#SBATCH --output=pstate_test_%j.out

IMAGE="${IMAGE:-lmsysorg/sglang:latest}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V3-0324}"
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.9}"
TP="${TP:-4}"

PRIMARY_HF_HOME="${HOME}/.cache/huggingface"
FALLBACK_HF_HOME="${HF_CACHE:-/mnt/data/home/ytian/.cache/huggingface}"

# Prefer the user's HuggingFace cache when present and writable; otherwise fall back.
if [ -d "${PRIMARY_HF_HOME}" ] && [ -w "${PRIMARY_HF_HOME}" ]; then
  HF_HOME="${PRIMARY_HF_HOME}"
  echo "Using HuggingFace cache at ${HF_HOME}"
else
  HF_HOME="${FALLBACK_HF_HOME}"
  echo "Primary HuggingFace cache unavailable; using fallback at ${HF_HOME}"
fi

mkdir -p "${HF_HOME}"

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker

echo "=== PSTATE TEST START: $(date -Iseconds) ==="
echo "Model: ${MODEL_PATH}"
echo "TP: ${TP}"
echo "Node: $(hostname)"

echo "=== CPU frequency check ==="
cat /sys/devices/system/cpu/amd_pstate/status 2>/dev/null || echo "amd_pstate status not available"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "scaling_governor not available"
grep "cpu MHz" /proc/cpuinfo | sort -t: -k2 -n | tail -5

echo "=== Starting SGLang server ==="
START_TIME=$(date +%s)

srun \
  --container-image="docker://${IMAGE}" \
  --container-mounts="${HF_HOME}:/root/.cache" \
  bash -c '
    export NCCL_DEBUG=WARN
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_SHM_DISABLE=0
    export NCCL_SOCKET_IFNAME=^lo,docker
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

    echo "=== Container start: $(date -Iseconds) ==="

    python3 -m sglang.launch_server \
      --model-path "'"${MODEL_PATH}"'" \
      --host "'"${HOST}"'" \
      --port "'"${PORT}"'" \
      --tp "'"${TP}"'" \
      --mem-fraction-static "'"${MEM_FRACTION_STATIC}"'"
  '

END_TIME=$(date +%s)
echo "=== PSTATE TEST END: $(date -Iseconds) ==="
echo "Total runtime: $((END_TIME - START_TIME)) seconds"
