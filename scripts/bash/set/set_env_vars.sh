#!/bin/bash


export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# the following PROJECT_POSTFIX hopefully would be the only variable to change for different projects
# MGFM-serving: the folder containing the Github repo for development
# MGFM: the parent folder used for deployment using PyCharm or a simple wrapper around the code repo
export PROJECT_POSTFIX="MGFM/MGFM-serving"

if hostname | grep -qi "discovery"; then
    # the usc carc discovery env
    export CLUSTER_NAME="discovery"
    export PROJECT_ACCOUNT="neiswang_1391"

    export HOME_PREFIX="/home1/$USER/workspace"
    export PROJECT_PREFIX="/project/${PROJECT_ACCOUNT}"
    export SCRATCH_PREFIX="/scratch1/$USER"
elif hostname | grep -qi "expanse"; then
    # the ucsd access expanse env
    export CLUSTER_NAME="expanse"
    export PROJECT_ACCOUNT="mia346"

    export HOME_PREFIX="/home/$USER/workspace"
    export PROJECT_PREFIX="/expanse/lustre/projects/${PROJECT_ACCOUNT}/$USER"
    export SCRATCH_PREFIX="/expanse/lustre/scratch/$USER/temp_project"
else
    # normal GPU instance, e.g., Vast
    export CLUSTER_NAME="None"
    export PROJECT_ACCOUNT="None"

    export HOME_PREFIX="/workspace"
    export PROJECT_PREFIX="/project"
    export SCRATCH_PREFIX="/scratch"
    mkdir -p "$PROJECT_PREFIX"
    mkdir -p "$SCRATCH_PREFIX"
fi

# debug usage
echo "Env name: ${CLUSTER_NAME}"

# set HF cache folders
export HF_HOME="${SCRATCH_PREFIX}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${SCRATCH_PREFIX}/.cache/huggingface/hub"
export HG_DATASETS_CACHE="${SCRATCH_PREFIX}/.cache/huggingface/datasets"

# set deepspeed, triton cache folder
export TRITON_CACHE_DIR="${SCRATCH_PREFIX}/.cache/triton_cache"

# index sbatch and slurm files within the project folder
export SLURM_PREFIX="./scripts/cluster/slurm"

export MODEL_CKPT_DIR="${PROJECT_PREFIX}/${PROJECT_POSTFIX}/model_ckpts"
export DATA_DIR="${PROJECT_PREFIX}/${PROJECT_POSTFIX}/datasets"
export OUTPUT_DIR="${PROJECT_PREFIX}/${PROJECT_POSTFIX}/outputs"
export CODE_DIR="${HOME_PREFIX}/${PROJECT_POSTFIX}"
