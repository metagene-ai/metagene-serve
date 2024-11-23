#!/bin/bash


## comment the following if in a slurm-based env
#source ./scripts/bash/set/set_env_vars.sh

export CKPT_STEP="step-00078000"

ST_MODEL_DIR="${MODEL_CKPT_DIR}/safetensors/${CKPT_STEP}"


# Quantize safetensors using NF4
NF4_CKPT_DIR="${MODEL_CKPT_DIR}/safetensors/nf4/${CKPT_STEP}"
mkdir -p "${NF4_CKPT_DIR}"
python ./src/format/quantize_st.py \
  --data_dir="${DATA_DIR}/sanity_check/cleaned_tokens_2000000000.txt" \
  --model_dir="${ST_MODEL_DIR}" \
  --quant_type="nf4" \
  --quant_model_dir="${NF4_CKPT_DIR}"
cp "${ST_MODEL_DIR}/tokenizer.model" "${NF4_CKPT_DIR}/tokenizer.model"
cp "${ST_MODEL_DIR}/tokenizer.json" "${NF4_CKPT_DIR}/tokenizer.json"


## Quantize safetensors using GPTQ
#GPTQ_CKPT_DIR="${MODEL_CKPT_DIR}/safetensors/gptq/${CKPT_STEP}"
#mkdir -p "${GPTQ_CKPT_DIR}"
#python ./src/format/quantize_st.py \
#  --data_dir="${DATA_DIR}/sanity_check/cleaned_tokens_2000000000.txt" \
#  --model_dir="${ST_MODEL_DIR}" \
#  --quant_type="gptq" \
#  --quant_model_dir="${GPTQ_CKPT_DIR}"
#cp "${ST_MODEL_DIR}/tokenizer.model" "${GPTQ_CKPT_DIR}/tokenizer.model"
#cp "${ST_MODEL_DIR}/tokenizer.json" "${GPTQ_CKPT_DIR}/tokenizer.json"
