#!/bin/bash


source ./serving/scripts/set/env/set_vars.sh

echo "Downloading litgpt model ckpt from Wasabi ..."

# export CKPT_STEP="step-00078000"
# REMOTE_MODEL_DIR="s3://mgfm-02/model-checkpoints/initial-checkpoints/${CKPT_STEP}"

export CKPT_STEP="step-00086000" # 00080000 <-> 00086000
REMOTE_MODEL_DIR="s3://mgfm-02/model-checkpoints/7b_part4_a/${CKPT_STEP}"
LOCAL_MODEL_DIR="${MODEL_CKPT_DIR}/litgpt/${CKPT_STEP}"
mkdir -p "${LOCAL_MODEL_DIR}"

aws s3 sync "${REMOTE_MODEL_DIR}" "${LOCAL_MODEL_DIR}" --endpoint-url=https://s3.us-west-1.wasabisys.com
echo "The litgpt model checkpoint has been downloaded to ${LOCAL_MODEL_DIR}"
