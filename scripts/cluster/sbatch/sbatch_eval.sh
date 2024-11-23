#!/bin/bash


# set base env variable based on GPU cluster
source ./scripts/bash/set/set_env_vars.sh

sbatch_script=$(basename "$0")
slurm_script=$(basename "$sbatch_script" | sed 's/^sbatch_//' | sed 's/\.sh$//').slurm
export SLURM_SCRIPT="${SLURM_PREFIX}/${slurm_script}"

########################## MAIN SCRIPT ##########################
export SBATCH_JOB_NAME="${OUTPUT_DIR}/eval"
export SBATCH_OUTPUT="${SBATCH_JOB_NAME}/%A_%a.out"

# wandb project env variable
export WANDB_API_KEY="3d8f78b6f45f87a40664beaf46a501b72faf205a" # make this private before git push
export WANDB_DIR="${OUTPUT_DIR}/wandb"

# LLM task-specific env variable
export FINETUNE_OUTPUT_DIR="${OUTPUT_DIR}/finetune/output"
export FINETUNE_LOG_DIR="${OUTPUT_DIR}/finetune/log"
########################## MAIN SCRIPT ##########################

# launch the slurm script
echo "Launching $SLURM_SCRIPT ..."
source ./scripts/cluster/sbatch/utils/sbatch_launch.sh
