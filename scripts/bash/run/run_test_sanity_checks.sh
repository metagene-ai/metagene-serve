#!/bin/bash


## comment the following if in a slurm-based env
#source ./scripts/bash/set/set_env_vars.sh

export CKPT_STEP="step-00078000"

#LIT_MODEL_DIR="${MODEL_CKPT_DIR}/litgpt/${CKPT_STEP}"
#pytest --log-level=INFO "${CODE_DIR}/src/test/test_sanity_checks.py" \
#  --model_format="litgpt" \
#  --model_dir="${LIT_MODEL_DIR}" \
#  --data_path="${DATA_DIR}/sanity_check/cleaned_tokens_2000000000.txt" \
#  --output_dir="${OUTPUT_DIR}/sanity_check/litgpt"

ST_MODEL_DIR="${MODEL_CKPT_DIR}/safetensors/${CKPT_STEP}"
echo "Running sanity check on safetensors model ..."
pytest -s "${CODE_DIR}/src/test/test_sanity_checks.py" \
  --model_format="st" \
  --model_dir="${ST_MODEL_DIR}" \
  --data_path="${DATA_DIR}/sanity_check/cleaned_tokens_2000000000.txt" \
  --output_dir="${OUTPUT_DIR}/sanity_check/safetensors"
