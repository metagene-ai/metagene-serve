#!/bin/bash
# python 3.12 + cuda 11.8.0 (Expanse or Vast) for vllm 0.6.6
# python 3.10 + cuda 11.8.0 (CARC) for vllm 0.6.4


export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
conda clean -a -y
mamba clean -a -y

# cuda, gcc/g++
conda install cuda -c nvidia/label/cuda-11.8.0 -y

# vLLM 0.6.6 with CUDA 11.8
pip install https://github.com/vllm-project/vllm/releases/download/v0.6.6/vllm-0.6.6+cu118-cp38-abi3-manylinux1_x86_64.whl

# mteb
MTEB_PATH="./serving/evaluate/gene-mteb"
if [ -d "${MTEB_PATH}" ]; then
    cd "${MTEB_PATH}" && git pull && cd -
else
    git clone https://github.com/shangshang-wang/gene-mteb.git "${MTEB_PATH}"
    cd "${MTEB_PATH}" && pip install -e . && cd -
fi