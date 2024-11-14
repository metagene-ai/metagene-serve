
# Change the following config for different experiment setup
#SBATCH --partition=gpu  ## gpu or gpu-shard
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1  ## 4 GPUs per node on Expanse
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00


