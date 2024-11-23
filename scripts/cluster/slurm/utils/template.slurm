#!/bin/bash


#SBATCH --account=$PROJECT_ACCOUNT
#SBATCH --job-name=$SBATCH_JOB_NAME
#SBATCH --output=$SBATCH_OUTPUT

# check the allocated GPUs
echo "Allocated GPU(s):"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Allocated GPU(s) memory:"
nvidia-smi --query-gpu=memory.total --format=csv,noheader

# activate the virtual env
eval "$(conda shell.bash hook)"
conda activate "${PROJECT_PREFIX}/envs/XXX"
echo "Currently activated Conda environment: $(basename "$CONDA_PREFIX")"

echo ""
nvcc --version
echo ""

# run the main file
#python xxx.py
#./xxx.sh