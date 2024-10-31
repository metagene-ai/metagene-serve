# MGFM-serving

Everything related to MGFM usage after training/tuning and inference-time optimization.

## Get Started

The following steps show an example workflow starting on a new vast instance.
```shell
sudo apt install gh
gh auth login
```

```shell
mkdir -p /workspace/MGFM
cd /workspace/MGFM
gh repo clone MetagenomicFM/MGFM-serving -- --recurse-submodules
cd MGFM-serving
find ./ -type f -name "*.sh" -exec chmod +x {} +
```

For the following scripts, please run them under the main repo folder rather their corresponding subfolders.
```shell
./submodules/vast-utilities/basic_setup/set_zsh_conda.sh
```

```shell
./submodules/vast-utilities/dev_setup/set_aws_wasabi.sh
./submodules/vast-utilities/dev_setup/get_model_ckpt.sh
```

```shell
./scripts/get_converted_models.sh
./scripts/get_finetune_setup.sh
./scripts/run_benchmark.sh
```
