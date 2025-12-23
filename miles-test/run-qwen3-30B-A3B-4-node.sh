#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint /host/home/Qwen3-30B-A3B
   #--hf-checkpoint /root/Qwen3-30B-A3B-FP8
   --ref-load /host/home/Qwen3-30B-A3B_torch_dist
   --load /host/home/Qwen3-30B-A3B_miles/
   --save /host/home/Qwen3-30B-A3B_miles/
   --save-interval 4
)

ROLLOUT_ARGS=(
   --prompt-data /host/home/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 4
   --eval-prompt-data aime /host/home/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.95
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   #--optimizer-cpu-offload
   #--overlap-cpu-optimizer-d2h-h2d
   #--use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-dev
   --wandb-group qwen3-30B-A3B-test
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MLP_WORKER_0_HOST}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
       "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
       "GLOO_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
       "TP_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
       "MASTER_ADDR": "${MLP_WORKER_0_HOST}",
       "PYTHONPATH": "/root/Megatron-LM/",
       "NCCL_CUMEM_ENABLE": "0",
       "CUDA_DEVICE_MAX_CONNECTIONS": "1",
       "NVTE_BWD_LAYERNORM_SM_MARGIN": "20",
       "NCCL_IB_TC": "160",
       "NCCL_PXN_DISABLE": "0",
       "NCCL_IB_GID_INDEX": "3",
       "NCCL_NET_GDR_LEVEL": "4",
       "NCCL_IB_RETRY_CNT": "7",
       "NCCL_IB_TIMEOUT": "32",
       "NCCL_IB_QPS_PER_CONNECTION": "8",
       "NCCL_P2P_LEVEL": "NVL",
       "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
       "NCCL_NVLS_ENABLE": "0",
       "NCCL_MIN_CTAS": "4",
       "OMPI_MCA_pml": "ob1",
       "OMPI_MCA_btl": "^openib",
       "OMPI_MCA_routed": "direct",
       "OMPI_MCA_routed_radix": "1024",
       "OMPI_MCA_plm_rsh_no_tree_spawn": "1",
       "OMPI_MCA_oob_tcp_if_include": "${MLP_SOCKET_IFNAME}",
       "OMPI_MCA_btl_tcp_if_include": "${MLP_SOCKET_IFNAME}"
       }
    }' \
   -- python3 train.py \
   --actor-num-nodes 4 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}