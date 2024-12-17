#!/bin/bash
# python 3.10 + cuda 11.8.0

# add the following to your .bashrc or running scripts
#export BNB_CUDA_VERSION=118
#export CUDA_HOME=$CONDA_PREFIX
#export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
conda clean -a -y
mamba clean -a -y

# cuda, gcc/g++, torch
conda install cuda -c nvidia/label/cuda-11.8.0 -y
mamba install pytorch=2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# bitsandbytes
mamba install bitsandbytes=0.45.0 -c conda-forge --no-pyc -y
pip install psutil transformers accelerate

# autoawq
pip install autoawq

# gptq from source to enable cuda extensions (need to be in a GPU env)
GPTQ_PATH="./serving/quantize/AutoGPTQ"
git clone https://github.com/PanQiWei/AutoGPTQ "${GPTQ_PATH}"
cd "${GPTQ_PATH}" && pip install -e . && cd -

# quanto
pip install optimum-quanto