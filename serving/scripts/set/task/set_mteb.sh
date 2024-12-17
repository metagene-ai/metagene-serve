#!/bin/bash
# python 3.10 + cuda 11.8.0


export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
conda clean -a -y
mamba clean -a -y

# cuda, gcc/g++, and torch
conda install cuda -c nvidia/label/cuda-11.8.0 -y
mamba install pytorch=2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# vLLM 0.6.4 with CUDA 11.8
pip install psutil sentencepiece py-cpuinfo transformers
pip install https://github.com/vllm-project/vllm/releases/download/v0.6.4/vllm-0.6.4+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install accelerate

# mteb
MTEB_PATH="./serving/evaluate/mteb"
git clone https://github.com/embeddings-benchmark/mteb.git "${MTEB_PATH}"
cd "${MTEB_PATH}" && pip install -e . && cd -