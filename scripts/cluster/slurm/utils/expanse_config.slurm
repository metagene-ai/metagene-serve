
# Change the following config for different experiment setup
## 3 GPU max on Exapnse gpu-shared with 30 mins waiting
## 4 GPU max on Exapnse gpu with XX mins waiting
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
