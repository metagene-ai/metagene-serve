#!/bin/bash


## Uncomment the following if it is not in a slurm-based env
#source ./scripts/bash/set/set_env_basic.sh

if [ -z "$CKPT_STEP" ] || [ "$CKPT_STEP" == "none" ]; then
    echo "Error: CKPT_STEP is not set or is set to 'none'. Exiting..."
    exit 1
fi

LIT_MODEL_DIR="${MODEL_CKPT_DIR}/litgpt/${CKPT_STEP}"
ST_MODEL_DIR="${MODEL_CKPT_DIR}/safetensors/${CKPT_STEP}"

#echo "Running sanity check on Litgpt model ..."
#python "${CODE_DIR}/src/evaluation/sanity_checks.py" \
#  --data_dir="${DATA_DIR}/sanity_check/cleaned_tokens_2000000000.txt" \
#  --model_dir="${LIT_MODEL_DIR}" \
#  --output_dir="${OUTPUT_DIR}/sanity_check/litgpt" \
#  --model_format="litgpt"

echo "Running sanity check on safetensors model ..."
python "${CODE_DIR}/src/evaluation/sanity_checks.py" \
  --data_dir="${DATA_DIR}/sanity_check/cleaned_tokens_2000000000.txt" \
  --model_dir="${ST_MODEL_DIR}" \
  --output_dir="${OUTPUT_DIR}/sanity_check/safetensors" \
  --model_format="st"
