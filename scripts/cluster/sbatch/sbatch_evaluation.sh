#!/bin/bash


# set base env variable based on GPU cluster
source ./scripts/bash/set/set_env_basic.sh

sbatch_file=$(basename "$0")
slurm_file=$(echo "$sbatch_file" | sed 's/^sbatch/slurm/')
export SLURM_SCRIPT="${SLURM_PREFIX}/${slurm_file}"
########################## MAIN SCRIPT ##########################
export SBATCH_JOB_NAME="${OUTPUT_DIR}/hp_opt"
export SBATCH_OUTPUT="${SBATCH_JOB_NAME}/%A_%a.out"

# wandb project env variable
export WANDB_API_KEY="3d8f78b6f45f87a40664beaf46a501b72faf205a" # make this private before git push
export WANDB_DIR="${SCRATCH_PREFIX}/wandb/projects/mgfm_serving"

# LLM task-specific env variable
export FINETUNE_OUTPUT_DIR="${SCRATCH_PREFIX}/mgfm_serving/output"
export FINETUNE_LOG_DIR="${SCRATCH_PREFIX}/mgfm_serving/log"
########################## MAIN SCRIPT ##########################

# launch the slurm script
echo "Launching $SLURM_SCRIPT ..."
source ./scripts/cluster/sbatch/utilities/sbatch_launch.sh
