#!/bin/bash


RUN_DIR=$(dirname "$(readlink -f "$0")")
CORE_DIR=$(dirname "$(dirname "$RUN_DIR")")
SET_DIR="${CORE_DIR}/scripts/set"

source "${SET_DIR}/env/set_vars.sh"
MAMBA_ENV="metagene-1"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"

CKPT_STEP="step-00078000"
ST_MODEL_DIR="${MODEL_CKPT_DIR}/safetensors"


# Quantize using bnb
export BNB_CUDA_VERSION=118
export CUDA_HOME="${CONDA_PREFIX}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

BNB_4BIT_CKPT_DIR="${ST_MODEL_DIR}/bnb-4bit"
BNB_8BIT_CKPT_DIR="${ST_MODEL_DIR}/bnb-8bit"
mkdir -p "${BNB_4BIT_CKPT_DIR}/${CKPT_STEP}"
mkdir -p "${BNB_8BIT_CKPT_DIR}/${CKPT_STEP}"
py_script="${CORE_DIR}/quantize/quantize_bnb.py"
echo "Running BNB quantization with safetensors ckpt ${CKPT_STEP} ..."
python "${py_script}" \
    --model_dir "${ST_MODEL_DIR}" \
    --model_dir_4bit "${BNB_4BIT_CKPT_DIR}" \
    --model_dir_8bit "${BNB_8BIT_CKPT_DIR}" \
    --model_ckpt "${CKPT_STEP}"


# Quantize using awq (8bit is not supported)
AWQ_4BIT_CKPT_DIR="${ST_MODEL_DIR}/awq-4bit"
mkdir -p "${AWQ_4BIT_CKPT_DIR}/${CKPT_STEP}"
py_script="${CORE_DIR}/quantize/quantize_awq.py"
echo "Running AWQ quantization with safetensors ckpt ${CKPT_STEP} ..."
python "${py_script}" \
    --model_dir "${ST_MODEL_DIR}" \
    --model_dir_4bit "${AWQ_4BIT_CKPT_DIR}" \
    --model_ckpt "${CKPT_STEP}"


# Quantize using quanto
QUANTO_4BIT_CKPT_DIR="${ST_MODEL_DIR}/quanto-4bit"
mkdir -p "${QUANTO_4BIT_CKPT_DIR}/${CKPT_STEP}"
py_script="${CORE_DIR}/quantize/quantize_quanto.py"
echo "Running QUANTO quantization with safetensors ckpt ${CKPT_STEP} ..."
python "${py_script}" \
    --model_dir "${ST_MODEL_DIR}" \
    --model_dir_4bit "${QUANTO_4BIT_CKPT_DIR}" \
    --model_ckpt "${CKPT_STEP}"