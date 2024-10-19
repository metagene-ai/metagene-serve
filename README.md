# MGFM-serving

Everything related to MGFM usage after training/tuning and inference-time optimization.

## Get Started

```shell
sudo apt install gh
gh auth login
```

```shell
cd /workspace
gh repo clone MetagenomicFM/MGFM-serving -- --recurse-submodules
cd ./MGFM-serving
find ./ -type f -name "*.sh" -exec chmod +x {} +
```

```shell
./vast-utilities/basic_setup/set_zsh_conda.sh
./vast-utilities/dev_setup/set_aws_wasabi.sh
./vast-utilities/dev_setup/get_model_ckpt.sh
```

```shell
./scripts/get_converted_models.sh
./scripts/get_quantized_models.sh
```
