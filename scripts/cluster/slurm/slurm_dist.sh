#!/bin/bash


#SBATCH --account=$PROJECT_ACCOUNT
#SBATCH --job-name=$SBATCH_JOB_NAME
#SBATCH --output=$SBATCH_OUTPUT

# # env variable for multi-gpu
# export WORLD_SIZE=2                           # Total number of processes (GPUs in use)
# export MASTER_ADDR=$(hostname)                # Set the master node address
# export MASTER_PORT=29500                      # Set an arbitrary but open port
# export RANK=0                                 # Rank of the node (0 for single node)
# export LOCAL_RANK=${SLURM_LOCALID}            # Local rank on this node (for multi-GPU)

# check the allocated GPUs
echo "Allocated GPU(s):"
nvidia-smi --query-gpu=name --format=csv,noheader

# run the python script
eval "$(conda shell.bash hook)"
conda activate ${PROJECT_PREFIX}/envs/test

# accelerate config
# accelerate launch --num_processes=2 ./src/test/gpu/test_accelerate.py
# torchrun --nproc_per_node=2 ./src/test/gpu/test_accelerate.py

# works in 4 and 8 bits with single GPU when combining accelerate and bitsandbytes
python ./src/test/gpu/test_accelerate.py
