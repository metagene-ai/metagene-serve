#!/bin/bash

#SBATCH --account=neiswang_1391
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          ## was 32
#SBATCH --gpus-per-task=1

## A100: epyc-7513|epyc-7513
## V100: xeon-6130
## A40: epyc-7313|epyc-7282
## P100: xeon-2640v4
#SBATCH --constraint=[epyc-7513|epyc-7513]
#SBATCH --time=5:00:00             ## hh:mm:ss
#SBATCH --array=1                 # specify <1-X>
#SBATCH --export=ALL
#SBATCH --output=/project/neiswang_1391/projects/MGFM/outputs/hp/%A_%a.out


eval "$(conda shell.bash hook)"
conda activate /project/neiswang_1391/envs/mgfm

echo "Allocated GPU(s):"
nvidia-smi --query-gpu=name --format=csv,noheader

python ./src/finetune/hyperparameter_opt.py
