#!/bin/bash


source ./scripts/bash/set/set_env_basic.sh
if [ -z "$CKPT_STEP" ] || [ "$CKPT_STEP" == "none" ]; then
    echo "Error: CKPT_STEP is not set or is set to 'none'. Exiting..."
    exit 1
fi

ST_MODEL_DIR="${MODEL_CKPT_DIR}/safetensors/${CKPT_STEP}"
NF4_CKPT_DIR="${MODEL_CKPT_DIR}/safetensors/nf4/${CKPT_STEP}"
#GPTQ_CKPT_DIR="${MODEL_CKPT_DIR}/safetensors/gptq/${CKPT_STEP}"

mkdir -p "${NF4_CKPT_DIR}"
#mkdir -p "${GPTQ_CKPT_DIR}"

# Quantize safetensors using NF4
python ./src/quantization/quantize_st.py \
  --data_dir="${DATA_DIR}/sanity_check/cleaned_tokens_2000000000.txt" \
  --model_dir="${ST_MODEL_DIR}" \
  --quant_type="nf4" \
  --quant_model_dir="${NF4_CKPT_DIR}"
cp "${ST_MODEL_DIR}/tokenizer.model" "${NF4_CKPT_DIR}/tokenizer.model"
cp "${ST_MODEL_DIR}/tokenizer.json" "${NF4_CKPT_DIR}/tokenizer.json"

### Quantize safetensors using GPTQ
#python ./src/quantization/quantize_st.py \
#  --data_dir="${DATA_DIR}/sanity_check/cleaned_tokens_2000000000.txt" \
#  --model_dir="${ST_MODEL_DIR}" \
#  --quant_type="gptq" \
#  --quant_model_dir="${GPTQ_CKPT_DIR}"
#cp "${ST_MODEL_DIR}/tokenizer.model" "${GPTQ_CKPT_DIR}/tokenizer.model"
#cp "${ST_MODEL_DIR}/tokenizer.json" "${GPTQ_CKPT_DIR}/tokenizer.json"
