#!/bin/bash


source ./mgfm.serving/scripts/set/set_env_vars.sh

# export CKPT_STEP="step-00078000"
#
# LIT_MODEL_DIR="${MODEL_CKPT_DIR}/litgpt/${CKPT_STEP}"
# pytest -s "${CODE_DIR}/src/test/test_sanity_checks.py" \
#     --sanity_check_model_format="litgpt" \
#     --sanity_check_model_dir="${LIT_MODEL_DIR}" \
#     --sanity_check_data_path="${DATA_DIR}/sanity_check/cleaned_tokens_2000000000.txt" \
#     --sanity_check_output_dir="${OUTPUT_DIR}/sanity_check/litgpt"
#
# ST_MODEL_DIR="${MODEL_CKPT_DIR}/safetensors/${CKPT_STEP}"
# echo "Running sanity check on safetensors model ..."
# pytest -s "${CODE_DIR}/src/test/test_sanity_checks.py" \
#     --sanity_check_model_format="st" \
#     --sanity_check_model_dir="${ST_MODEL_DIR}" \
#     --sanity_check_data_path="${DATA_DIR}/sanity_check/cleaned_tokens_2000000000.txt" \
#     --sanity_check_output_dir="${OUTPUT_DIR}/sanity_check/safetensors"
