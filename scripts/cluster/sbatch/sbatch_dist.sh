#!/bin/bash


# set base env variable based on GPU cluster
source ./scripts/cluster/sbatch/utilities/sbatch_base.sh
sbatch_file=$(basename "$0")
slurm_file=$(echo "$sbatch_file" | sed 's/^sbatch/slurm/')
export SLURM_SCRIPT="${SLURM_PREFIX}/${slurm_file}"

########################## MAIN SCRIPT ##########################
export SBATCH_JOB_NAME="projects/mgfm_serving/test/dist"
export SBATCH_OUTPUT="${PROJECT_PREFIX}/${SBATCH_JOB_NAME}/%A_%a.out"

export BNB_CUDA_VERSION=118

export HF_HOME="${SCRATCH_PREFIX}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${SCRATCH_PREFIX}.cache/huggingface/hub"
export HG_DATASETS_CACHE="${SCRATCH_PREFIX}/.cache/huggingface/datasets"

export OFFLOAD_FOLDER="${SCRATCH_PREFIX}/offload_folder"
mkdir -p $OFFLOAD_FOLDER
########################## MAIN SCRIPT ##########################

# launch the slurm script
echo "Launching $SLURM_SCRIPT ..."
source ./scripts/cluster/sbatch/utilities/sbatch_launch.sh
