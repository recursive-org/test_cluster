Testing miles training on Andromeda cluster mostly follow the miles' quick start guide

# Prerequisites
- Andromeda cluster access
- Clone miles repo to the persistent volume
- Download model and datasets
```
pip install -U huggingface_hub

# Download model weights (GLM-Z1-9B) for single node example
hf download zai-org/GLM-Z1-9B-0414 --local-dir /local/path/to/GLM-Z1-9B-0414

# Download model weights (Qwen3-30B-A3B) for multi node example
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir /local/path/to/Qwen3-30B-A3B

# Download training dataset (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /local/path/to/dapo-math-17k

# Download evaluation dataset (aime-2024)
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /local/path/to/aime-2024
```

# Interactive mode test
- Login to the head node of the cluster
- Open a tmux session (Optional, recommended so that you don't accidentally lose the job)

## Single Node training of GLM-9B
- Run
```
srun -C gpu --gpus=8 --pty --mem=512G --no-container-mount-home --container-image=radixark/miles:latest --container-mounts=/mnt/data/home/ybzhou:/host/home --network=host bash -i
```
Note:
I have tested mem=512G and above, you definitely need more than 128G RAM

The miles image have things in the `/root` directory, and it seems Andromeda automatically mounts the persistent vo;ume under root, so `--no-container-mount-home` is required

`--container-mounts=src:dest` can specify the desired mount points in dest (i.e inside the container)

- Once inside the compute node, clone the miles repo to the mounted persistent volume (e.g. `/host/home/git/miles` in this example), install the library
```
cd /host/home/git/miles
pip install -e .
```

- Setting some environment variables
```
export NCCL_SOCKET_IFNAME=ens26np0
export GLOO_SOCKET_IFNAME=ens26np0
export WANDB_KEY=<your wandb key>
```

the `ens26np0` is the network interface name, you can find it by `ip -br addr`, pick one that is connected to the network

- Convert weights
```
cd /host/home/git/miles
source scripts/models/glm4-9B.sh
```

Do the actual conversion
```
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /host/home/GLM-Z1-9B-0414 \
    --save /host/home/GLM-Z1-9B-0414_torch_dist
```

Note: PYTHONPATH uses `/root/Megatron-LM` which is inside the container

- run training
```
cd /host/home/git/miles
bash scripts/run-glm4-9B.sh
```

if you would like to use wandb to track training, uncomment/modify the  `WANDB_ARGS`  block in the `run-glm4-9B.sh` file

See more at https://github.com/radixark/miles/blob/main/docs/en/get_started/quick_start.md for explainations of arguments in the .sh file

## Multi Node (4 Nodes Example)
- Allocate nodes
```
salloc --gres=gpu:b200:8 -N 4 --ntasks-per-node=1 --cpus-per-task=32 --time=12:00:00 --mem=1024GB
```

- Check and note down the allocated nodes
```
scontrol show hostnames "$SLURM_NODELIST"
```

- Check and note down the jobid
```
squeue -u $USER
```

- Login to each of the compute node
```
# login to head node
srun --jobid=<jobid> --nodes=1 --ntasks=1 --gres=gpu:8  --nodelist=<node1-name> --no-container-mount-home --container-image=radixark/miles:latest --container-mounts=/mnt/data/home/ybzhou:/host/home --network=host --pty bash

# login to worker node 2
srun --jobid=<jobid> --nodes=1 --ntasks=1 --gres=gpu:8 --nodelist=<node2-name> --no-container-mount-home --container-image=radixark/miles:latest --container-mounts=/mnt/data/home/ybzhou:/host/home --network=host --pty bash

# login to worker node 3
srun --jobid=<jobid> --nodes=1 --ntasks=1 --gres=gpu:8 --nodelist=<node3-name> --no-container-mount-home --container-image=radixark/miles:latest --container-mounts=/mnt/data/home/ybzhou:/host/home --network=host --pty bash

# login to worker node 4
srun --jobid=<jobid> --nodes=1 --ntasks=1 --gres=gpu:8 --nodelist=<node4-name> --no-container-mount-home --container-image=radixark/miles:latest --container-mounts=/mnt/data/home/ybzhou:/host/home --network=host --pty bash

```

- Note down their IP Address
```
ip -br addr
```

- Convert weights
```
cd /host/home/git/miles
source scripts/models/qwen3-30B-A3B.sh
```

Do the conversion
```
PYTHONPATH=/root/Megatron-LM/ torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /host/home/Qwen3-30B-A3B/ \
   --save /host/home/Qwen3-30B-A3B_torch_dist/
```

- Place `run-qwen3-30B-A3B-4-node.sh` in the same directory as `run-qwen3-30B-A3B.sh`, i.e. `scripts/`

- Set some environment variables
```
export NCCL_SOCKET_IFNAME=ens26np0
export GLOO_SOCKET_IFNAME=ens26np0
export MLP_SOCKET_IFNAME=ens26np0
export WANDB_KEY=<your wandb key>
export MLP_WORKER_0_HOST=<node1-ip>
```

- We use node1 as the master node and rest as worker nodes
log into node1 and run
```
bash scripts/run-qwen3-30B-A3B-4-node.sh
```

log into other nodes and run
```
ray start --address=<node1-ip>:6379 --num-gpus 8 --node-ip-address <node-ip> --disable-usage-stats
```

# Non-interactive mode test

## Single Node training of GLM-9B
run
```
sbatch run-glm4-9B.sh
```

Use `tail -f miles-glm9b-training-<jobid>.out` to check the training log


## Multi Node (4 Nodes Example)
