#!/bin/bash


## Uncomment the following if it is not in a slurm-based env
source ./scripts/bash/set/set_env_basic.sh

if [ -z "$CKPT_STEP" ] || [ "$CKPT_STEP" == "none" ]; then
    echo "Error: CKPT_STEP is not set or is set to 'none'. Exiting..."
    exit 1
fi

echo "Star converting Litgpt model to HF model ..."

ORIGINAL_MODEL_DIR="${MODEL_CKPT_DIR}/litgpt/${CKPT_STEP}"
PTH_MODEL_DIR="${MODEL_CKPT_DIR}/pth/${CKPT_STEP}"
ST_MODEL_DIR="${MODEL_CKPT_DIR}/safetensors/${CKPT_STEP}"

mkdir -p "${PTH_MODEL_DIR}"
mkdir -p "${ST_MODEL_DIR}"

# Convert litgpt format to pth format
echo "First, converting litgpt model to pth ..."
litgpt convert_from_litgpt "${ORIGINAL_MODEL_DIR}" "${PTH_MODEL_DIR}"

cp "${ORIGINAL_MODEL_DIR}/tokenizer.model" "${PTH_MODEL_DIR}/tokenizer.model"
cp "${ORIGINAL_MODEL_DIR}/config.json" "${PTH_MODEL_DIR}/config.json"
cp "${ORIGINAL_MODEL_DIR}/tokenizer.json" "${PTH_MODEL_DIR}/tokenizer.json"

# Convert pth format to safetensors format
python "${CODE_DIR}/src/conversion/convert_pth_to_st.py" \
    --pth_model_dir="${PTH_MODEL_DIR}" \
    --st_model_dir="${ST_MODEL_DIR}"

cp "${ORIGINAL_MODEL_DIR}/tokenizer.model" "${ST_MODEL_DIR}/tokenizer.model"
cp "${ORIGINAL_MODEL_DIR}/config.json" "${ST_MODEL_DIR}/config.json"
cp "${ORIGINAL_MODEL_DIR}/tokenizer.json" "${ST_MODEL_DIR}/tokenizer.json"

# clean up pth model
rm -rf "${PTH_MODEL_DIR}"
