#!/bin/bash


# set base env variable based on GPU cluster
source ./scripts/bash/set/set_env_vars.sh

sbatch_script=$(basename "$0")
slurm_script=$(basename "$sbatch_script" | sed 's/^sbatch_//' | sed 's/\.sh$//').slurm
export SLURM_SCRIPT="${SLURM_PREFIX}/${slurm_script}"

########################## MAIN SCRIPT ##########################
export SBATCH_JOB_NAME="${OUTPUT_DIR}/test_xxx" # change test_xxx to different test cases
export SBATCH_OUTPUT="${SBATCH_JOB_NAME}/%A_%a.out"
########################## MAIN SCRIPT ##########################

# launch the slurm script
echo "Launching $SLURM_SCRIPT ..."
source ./scripts/cluster/sbatch/utils/sbatch_launch.sh
