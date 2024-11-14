
# Change the following config for different experiment setup
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1  ## 2 GPUs per node on CARC
#SBATCH --time=1:00:00
#SBATCH --array=1
#SBATCH --export=ALL

## A100: epyc-7513 & A40: epyc-7313|epyc-7282 & P100: xeon-2640v4 & V100: xeon-6130
#SBATCH --constraint=[epyc-7513|epyc-7313|epyc-7282]
