#!/bin/bash
#SBATCH --job-name=miles-multinode-training
#SBATCH -C gpu
#SBATCH --gres=gpu:b200:8
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=3-04:00:00
#SBATCH --mem=1024GB
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

# Container and Mounts (Update if necessary)
# Note: Ensure /your/data/path matches your actual home dir mount on the cluster if needed
CONTAINER_IMAGE="radixark/miles:latest"
CONTAINER_MOUNTS="/your/data/path:/host/home"

# Project Directory
PROJECT_DIR="/host/home/git/miles"

# Node List File
NODE_LIST_FILE="nodes.txt"

# -------------------------------------------------------------------------
# 1. Generate/Read Node List
# -------------------------------------------------------------------------

echo "Job started on $(hostname)"

if [ ! -f "$NODE_LIST_FILE" ]; then
    echo "Node list file '$NODE_LIST_FILE' not found. Generating from SLURM_NODELIST..."
    scontrol show hostnames "$SLURM_NODELIST" > "$NODE_LIST_FILE"
else
    echo "Using existing node list file: $NODE_LIST_FILE"
fi

# Read nodes into array
mapfile -t NODES < "$NODE_LIST_FILE"

if [ ${#NODES[@]} -eq 0 ]; then
    echo "Error: No nodes found in $NODE_LIST_FILE"
    exit 1
fi

HEAD_NODE="${NODES[0]}"
WORKER_NODES=("${NODES[@]:1}")

echo "Head Node: $HEAD_NODE"
echo "Worker Nodes: ${WORKER_NODES[*]}"

# -------------------------------------------------------------------------
# 2. Setup Environment Variables
# -------------------------------------------------------------------------

# Get Head Node IP
# We can just resolve it from the hostname command executed on the head node via ssh, 
# or use srun if simple resolution isn't enough. 
# Here we'll ssh to head node and get its IP.
HEAD_NODE_IP=$(ssh "$HEAD_NODE" "hostname -I | awk '{print \$1}'")
echo "Head Node IP: $HEAD_NODE_IP"

# Common Environment Variables to export to all sessions
export GLOBAL_ENV_VARS="export NCCL_SOCKET_IFNAME=ens26np0; export GLOO_SOCKET_IFNAME=ens26np0; export MLP_SOCKET_IFNAME=ens26np0; export MLP_WORKER_0_HOST=$HEAD_NODE_IP; export WANDB_KEY=${WANDB_KEY}"

# -------------------------------------------------------------------------
# 3. Launch Jobs via SSH
# -------------------------------------------------------------------------

# --- Head Node ---
echo "Starting Head Node process on $HEAD_NODE..."
# Environment Setup
$GLOBAL_ENV_VARS

# Run Container
# Note: We need to run inside the container. The README example for interactive mode 
# uses 'srun ... bash', then runs commands inside.
# Since we are automating, we will construct a docker/srun command or simply assume 
# we can run the script if the environment is already set up or if we use srun to launch the container.

# Re-constructing the srun command from README but for non-interactive execution:
srun --nodes=1 --ntasks=1 --gres=gpu:8 --nodelist=${HEAD_NODE} \
        --no-container-mount-home \
        --container-image=${CONTAINER_IMAGE} \
        --container-mounts=${CONTAINER_MOUNTS} \
        --network=host \
        bash -c "
        cd ${PROJECT_DIR}
        pip install -e .
        ${GLOBAL_ENV_VARS}
        # Run the head node script
        bash scripts/run-qwen3-30B-A3B-4-node.sh" &

# Give the head node a moment to initialize
sleep 100

# --- Worker Nodes ---
for WORKER_NODE in "${WORKER_NODES[@]}"; do
    echo "Starting Worker Node process on $WORKER_NODE..."
    # Environment Setup
    $GLOBAL_ENV_VARS
    
    # Run Container via srun
    srun --nodes=1 --ntasks=1 --gres=gpu:8 --nodelist=$WORKER_NODE \
            --no-container-mount-home \
            --container-image=$CONTAINER_IMAGE \
            --container-mounts=$CONTAINER_MOUNTS \
            --network=host \
            bash -c "
            cd ${PROJECT_DIR} && pip install -e .
            ${GLOBAL_ENV_VARS}
            
            # Run Ray Worker
            ray start --address=$HEAD_NODE_IP:6379 --num-gpus 8 --node-ip-address \$(hostname -I | awk '{print \$1}') --disable-usage-stats --block" &
done

wait