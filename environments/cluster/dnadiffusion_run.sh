#!/usr/bin/env bash
#SBATCH --job-name=openbioml
#SBATCH --account openbioml
#SBATCH --partition=g40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --output=./logs/dnadiffusion_%A_%a.out
#SBATCH --cpus-per-gpu=12

# set -euxo pipefail
set -x

# https://askubuntu.com/a/1389915/21876
[ ! -f .env ] || source .env

# https://unix.stackexchange.com/q/569988/9185
export | cut -d" " -f3-

echo "$PWD"

export NCCL_PROTO=simple
export NCCL_DEBUG=WARN
export NCCL_TREE_THRESHOLD=0
export NCCL_IBEXT_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64

export PYTHONFAULTHANDLER=1

export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_btl="^openib"

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export WORLD_SIZE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export JOB_COMMENT="Key=Monitoring,Value=ON"

echo myuser = $(whoami)
echo COUNT_NODE="$COUNT_NODE"
echo LD_LIBRARY_PATH = "$LD_LIBRARY_PATH"
echo PATH = "$PATH"
echo which mpicc $(which mpicc)
echo HOSTNAMES = "$HOSTNAMES"
echo hostname = $(hostname)
echo MASTER_ADDR= "$MASTER_ADDR"
echo MASTER_PORT= "$MASTER_PORT"

H=$(hostname)
THEID=$(echo -e "$HOSTNAMES"  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]")
echo THEID="$THEID"

echo go COUNT_NODE="$COUNT_NODE"
echo MASTER_ADDR="$MASTER_ADDR"
echo MASTER_PORT="$MASTER_PORT"
echo WORLD_SIZE="$WORLD_SIZE"

source "$WORK_HOME"/"$CONDATYPE"/bin/activate "$CONDA_ENVIRONMENT"

cd "$DNADIFFUSION_HOME" || exit

# https://gimmemotifs.readthedocs.io/en/master/overview.html#running-on-a-cluster
export XDG_CACHE_HOME=$(mktemp -d)
echo "Using $XDG_CACHE_HOME for gimmemotifs cache"
echo "$PWD"

accelerate launch  \
    --num_processes $(( 8 * $COUNT_NODE )) \
    --num_machines "$COUNT_NODE" \
    --machine_rank "$THEID" \
    --multi_gpu \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --mixed_precision 'no' \
    --dynamo_backend 'no' \
    train.py
