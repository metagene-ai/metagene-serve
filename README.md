# mgfm.serving

Everything related to MGFM usage after training/tuning and inference-time optimization.

## Working Sheet

* ✅ Litgpt to HF, gguf model conversion
* ✅ Rebuild HF tokenizer
* ✅ New version of sanity check
* ✅ Bnb, gptq, awq quantization 
* 🧑🏽‍💻 Get access to Jason's mteb fork and add it as a submodule
* 🧑🏽‍💻 Ask willie to upload the tokenizer model to wasabi litgpt model for archive 

## Quick Tour

Sure all the scripts are executable:
```shell
find ./ -type f \( -name "*.sh" -o -name "*.slurm" \) -exec chmod +x {} +
```

Request GPU in a slurm-based environment:
```shell
## A100: epyc-7513|a100-40gb|a100-80gb & A40: epyc-7313|epyc-7282 & P100: xeon-2640v4 & V100: xeon-6130
srun --partition=gpu --constraint=[epyc-7513|epyc-7313|epyc-7282] --account=neiswang_1391 --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=2:00:00 --export=ALL --pty bash -i
```
