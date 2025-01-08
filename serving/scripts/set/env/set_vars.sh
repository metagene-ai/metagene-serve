#!/bin/bash


export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1

export DS_LOG_LEVEL=error
export MKL_THREADING_LAYER=GNU


if hostname | grep -qi "usc"; then
    # the USC CARC Discovery env
    export CLUSTER_NAME="usc discovery"
    echo "Env name: ${CLUSTER_NAME}"
    export PROJECT_ACCOUNT="neiswang_1391"

    export HOME_PREFIX="/home1/$USER/workspace"
    export PROJECT_PREFIX="/project/${PROJECT_ACCOUNT}"
    export SCRATCH_PREFIX="/scratch1/$USER"
else
    # the ACCESS SDSC Expanse env
    export CLUSTER_NAME="sdsc expanse"
    echo "Env name: ${CLUSTER_NAME}"
    export PROJECT_ACCOUNT="wis189"

    export HOME_PREFIX="/home1/$USER/workspace"
    export PROJECT_PREFIX="/expanse/lustre/projects/${PROJECT_ACCOUNT}/$USER/projects"
    export SCRATCH_PREFIX="/expanse/lustre/scratch/$USER/temp_project/projects"
fi

# # common GPU instance, e.g., Vast and Primeintellect
# export CLUSTER_NAME="GPU instance"
# echo "Env name: ${CLUSTER_NAME}"
# export PROJECT_ACCOUNT="None"

# export HOME_PREFIX="/workspace"
# export PROJECT_PREFIX="/project"
# export SCRATCH_PREFIX="/scratch"
# mkdir -p "$PROJECT_PREFIX"
# mkdir -p "$SCRATCH_PREFIX"

# the following PROJECT_POSTFIX hopefully would be the only variable to change for different projects
export SUBPROJET_POSTFIX="serve"
export PROJECT_POSTFIX="metagene/metagene-${SUBPROJET_POSTFIX}"
export PYTHONPATH="${HOME_PREFIX}/${PROJECT_POSTFIX}":$PYTHONPATH
export PYTHONPATH="${HOME_PREFIX}/${PROJECT_POSTFIX}/${SUBPROJET_POSTFIX}":$PYTHONPATH

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
export HF_TOKEN="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"

export TRITON_CACHE_DIR="${CACHE_DIR}/triton_cache"

export WANDB_API_KEY="8a590118879d8c43eac0ebb53bea5bdd437e87c8"
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}/wandb"

export BNB_CUDA_VERSION=118
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"