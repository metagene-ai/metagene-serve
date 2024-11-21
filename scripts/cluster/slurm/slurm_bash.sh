#!/bin/bash


#SBATCH --account=$PROJECT_ACCOUNT
#SBATCH --job-name=$SBATCH_JOB_NAME
#SBATCH --output=$SBATCH_OUTPUT


# check the allocated GPUs
echo "Allocated GPU(s):"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Allocated GPU(s) memory:"
nvidia-smi --query-gpu=memory.total --format=csv,noheader

# run the python script
eval "$(conda shell.bash hook)"
conda activate "${PROJECT_PREFIX}/envs/mgfm-serving"
echo "Currently activated Conda environment: $(basename "$CONDA_PREFIX")"

echo ""
nvcc --version
echo ""

export CKPT_STEP="step-00078000"
source ./scripts/bash/run/run_sanity_checks.sh
