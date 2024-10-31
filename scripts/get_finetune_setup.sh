#!/bin/bash

pip install gdown
pip install wandb
pip install transformers
pip install torch
pip install scikit-learn
pip install accelerate

## Enable flash attention
#gh repo clone triton-lang/triton
#cd triton/python
#pip install cmake
#pip install -e .

# Get the gdown path for fine-tune data
get_input() {
    read -p "$1: " value
    echo $value
}
gdown_path=$(get_input "Enter your remote gdown path for finetune dataset")

mkdir -p /workspace/MGFM/data/fine-tune

gdown $gdown_path -O /workspace/MGFM/data/fine-tune/

cd 
unzip -q /workspace/MGFM/data/fine-tune/GUE.zip -d /workspace/MGFM/data/fine-tune/
rm /workspace/MGFM/data/fine-tune/GUE.zip 
