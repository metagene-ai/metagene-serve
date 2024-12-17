#!/bin/bash


export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=true

if hostname | grep -qi "usc"; then
    # the USC CARC Discovery env
    export CLUSTER_NAME="usc discovery"
    echo "Env name: ${CLUSTER_NAME}"
    export PROJECT_ACCOUNT="neiswang_1391"

    export HOME_PREFIX="/home1/$USER/workspace"
    export PROJECT_PREFIX="/project/${PROJECT_ACCOUNT}"
    export SCRATCH_PREFIX="/scratch1/$USER"
else
    # common GPU instance, e.g., Vast and Primeintellect
    export CLUSTER_NAME="GPU instance"
    echo "Env name: ${CLUSTER_NAME}"
    export PROJECT_ACCOUNT="None"

    export HOME_PREFIX="/workspace"
    export PROJECT_PREFIX="/project"
    export SCRATCH_PREFIX="/scratch"
    mkdir -p "$PROJECT_PREFIX"
    mkdir -p "$SCRATCH_PREFIX"
fi

# the following PROJECT_POSTFIX hopefully would be the only variable to change for different projects
# MGFM-serving: the folder containing the Github repo for development
# MGFM: the parent folder used for deployment using PyCharm or a simple wrapper around the code repo
export PROJECT_POSTFIX="MGFM/MGFM-serving"
export PYTHONPATH="${HOME_PREFIX}/${PROJECT_POSTFIX}":$PYTHONPATH
export PYTHONPATH="${HOME_PREFIX}/${PROJECT_POSTFIX}/serving":$PYTHONPATH

export PROJECT_DIR="${PROJECT_PREFIX}/${PROJECT_POSTFIX}"
export ENV_DIR="${PROJECT_PREFIX}/envs"
export MODEL_CKPT_DIR="${PROJECT_DIR}/model_ckpts"
export DATA_DIR="${PROJECT_DIR}/datasets"
export OUTPUT_DIR="${PROJECT_DIR}/outputs"
export LOGGING_DIR="${PROJECT_DIR}/logging"

export CACHE_DIR="${PROJECT_DIR}/.cache"
export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/huggingface/hub"
export HG_DATASETS_CACHE="${CACHE_DIR}/huggingface/datasets"
export TRITON_CACHE_DIR="${CACHE_DIR}/triton_cache"