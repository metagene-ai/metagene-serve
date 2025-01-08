# METAGENE-1: Metagenomic Foundation Model for Pandemic Monitoring

This repository contains serving code for METAGENE-1.

## Quick Tour

Sure all the scripts are executable:
```shell
find ./ -type f \( -name "*.sh" -o -name "*.slurm" \) -exec chmod +x {} +
```

Request GPU in a slurm-based environment:
```shell
# On CARC
## A100: epyc-7513|a100-40gb|a100-80gb & A40: epyc-7313|epyc-7282 & P100: xeon-2640v4 & V100: xeon-6130
srun --partition=gpu --constraint=epyc-7313 --account=neiswang_1391 --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=5:00:00 --export=ALL --pty bash -i

# On Expanse
srun --partition=gpu --account=wis189 --nodes=1 --ntasks-per-node=1 --gpus=4 --cpus-per-task=16 --mem=256G --time=8:00:00 --export=ALL --pty bash -i
```
