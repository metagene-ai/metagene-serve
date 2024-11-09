# MGFM-serving

Everything related to MGFM usage after training/tuning and inference-time optimization.

## Get Started

The following steps show an example workflow starting on a new env.
```shell
sudo apt install gh
gh auth login

mkdir -p ~/workspace/MGFM && cd ~/workspace/MGFM
gh repo clone MetagenomicFM/MGFM-serving -- --recurse-submodules
cd MGFM-serving
```

```shell
pip install -r ./requirements/requirement-basic.txt
pip install -r ./requirements/requirement-finetune.txt
```

```shell
find ./ -type f -name "*.sh" -exec chmod +x {} +
./submodules/server-utilities/basic_setup/set_conda.sh
./submodules/server-utilities/dev_setup/set_aws_wasabi.sh
```

For fine-tuning and evaluation tasks, please run the following
```shell
./scripts/get_finetune_setup.sh
./run/run_benchmark_eval.sh
```
