#!/bin/bash
#SBATCH --job-name=miles-glm9b-training
#SBATCH -C gpu
#SBATCH --gpus=8
#SBATCH --mem=512G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=4:00:00

# Container configuration based on README
CONTAINER_IMAGE="radixark/miles:latest"
# WARNING: Update the source path below to your actual home directory on the cluster if it is not /mnt/data/home/ybzhou
CONTAINER_MOUNTS="/mnt/data/home/ybzhou:/host/home"

# Environment Setup
# Note: The README specifies --no-container-mount-home and --network=host

echo "Starting job on node: $HOSTNAME"
echo "Date: $(date)"

srun --no-container-mount-home \
     --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${CONTAINER_MOUNTS}" \
     --network=host \
     bash -c "
set -e # Exit immediately if a command exits with a non-zero status

# 1. Install miles
# 'Once inside the compute node, clone the miles repo to the mounted persistent volume (e.g. /host/home/git/miles in this example), install the library'
# Attempting to cd to the directory specified in the README
if [ -d \"/host/home/git/miles\" ]; then
    cd /host/home/git/miles
    echo \"Installing miles from $(pwd)...\"
    pip install -e .
else
    echo \"Error: Directory /host/home/git/miles does not exist inside container.\"
    echo \"Please check your mount paths.\"
    exit 1
fi

# 2. Set environment variables
echo \"Setting environment variables...\"
export NCCL_SOCKET_IFNAME=ens26np0
export GLOO_SOCKET_IFNAME=ens26np0
# export WANDB_KEY=<your wandb key> # Uncomment and set if needed

# 3. Convert weights
echo \"Converting weights...\"
source scripts/models/glm4-9B.sh

# The README runs this python script with PYTHONPATH set
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    \${MODEL_ARGS[@]} \
    --hf-checkpoint /host/home/GLM-Z1-9B-0414 \
    --save /host/home/GLM-Z1-9B-0414_torch_dist

# 4. Run training
echo \"Running training...\"
bash scripts/run-glm4-9B.sh
"
