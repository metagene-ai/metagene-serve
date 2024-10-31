#!/bin/bash

pip install gdown

# Get the gdown path for fine-tune data
get_input() {
    read -p "$1: " value
    echo $value
}
gdown_path=$(get_input "Enter your remote gdown path for finetune dataset")

mkdir -p /workspace/MGFM/data/fine-tune

gdown $gdown_path -O /workspace/MGFM/data/fine-tune/

filename=$(ls /workspace/MGFM/data/fine-tune/ -t *.zip | head -n1)
unzip -q $filename
rm $filename
