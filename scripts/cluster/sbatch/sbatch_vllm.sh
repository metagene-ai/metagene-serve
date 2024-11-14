#!/bin/bash


# set base env variable based on GPU cluster
source ./scripts/cluster/sbatch/utilities/sbatch_base.sh
sbatch_file=$(basename "$0")
slurm_file=$(echo "$sbatch_file" | sed 's/^sbatch/slurm/')
export SLURM_SCRIPT="${SLURM_PREFIX}/${slurm_file}"

########################## MAIN SCRIPT ##########################
export SBATCH_JOB_NAME="projects/mgfm_serving/inference/vllm"
export SBATCH_OUTPUT="${PROJECT_PREFIX}/${SBATCH_JOB_NAME}/%A_%a.out"
########################## MAIN SCRIPT ##########################

# launch the slurm script
echo "Launching $SLURM_SCRIPT ..."
source ./scripts/cluster/sbatch/utilities/sbatch_launch.sh