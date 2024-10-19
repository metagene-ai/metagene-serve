#!/bin/bash

pip install gdown

# Get the gdown path for sanity check data
get_input() {
    read -p "$1: " value
    echo $value
}
gdown_path=$(get_input "Enter your remote gdown path for dataset")

mkdir -p /workspace/MGFM/data/sanity_check

gdown $gdown_path -O /workspace/MGFM/data/sanity_check/
