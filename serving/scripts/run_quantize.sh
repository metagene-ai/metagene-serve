#!/bin/bash


CONDA_ENV="metagene"
eval "$(conda shell hook --shell bash)" && conda activate "${CONDA_ENV}"

# Quantize using bnb
export BNB_CUDA_VERSION=118
export CUDA_HOME="${CONDA_PREFIX}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

echo "Running BNB quantization ..."
python ./serving/quantize/quantize_bnb.py

# Quantize using awq
echo "Running AWQ quantization ..."
python ./serving/quantize/quantize_awq.py

# Quantize using quanto
echo "Running QUANTO quantization ..."
python ./serving/quantize/quantize_quanto.py