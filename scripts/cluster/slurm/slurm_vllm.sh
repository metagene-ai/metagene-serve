#!/bin/bash


#SBATCH --account=$PROJECT_ACCOUNT
#SBATCH --job-name=$SBATCH_JOB_NAME
#SBATCH --output=$SBATCH_OUTPUT

# check the allocated GPUs
echo "Allocated GPU(s):"
nvidia-smi --query-gpu=name --format=csv,noheader

# run the python script
eval "$(conda shell.bash hook)"
conda activate ${PROJECT_PREFIX}/envs/mgfm-serving
python ./src/test/test_vllm.py
