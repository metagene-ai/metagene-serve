#!/bin/bash


# Must start a fresh new conda env! https://docs.vllm.ai/en/stable/getting_started/installation.html
# Install vLLM with CUDA 11.8.
export VLLM_VERSION=0.6.1.post1
export PYTHON_VERSION=310
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118

# Enable flash attention 2
pip install vllm-flash-attn

# TODO change this to dev install
pip install mteb
