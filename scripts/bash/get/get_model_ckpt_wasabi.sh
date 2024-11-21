#!/bin/bash


source ./scripts/bash/set/set_env_basic.sh

export CKPT_STEP="step-00078000"
#export CKPT_STEP="step-00080000"

echo "Downloading model checkpoint from Wasabi ..."
REMOTE_MODEL_DIR=s3://mgfm-02/model-checkpoints/initial-checkpoints/step-00078000
#REMOTE_MODEL_DIR="s3://mgfm-02/model-checkpoints/7b_part4_a/${CKPT_STEP}"
#REMOTE_MODEL_DIR="s3://mgfm-02/model-weights/hf-ckpt-78k"

LOCAL_MODEL_DIR="${MODEL_CKPT_DIR}/litgpt/${CKPT_STEP}"
#LOCAL_MODEL_DIR="${MODEL_CKPT_DIR}/safetensors/${CKPT_STEP}"
mkdir -p "${LOCAL_MODEL_DIR}"

aws s3 sync "${REMOTE_MODEL_DIR}" "${LOCAL_MODEL_DIR}" --endpoint-url=https://s3.us-west-1.wasabisys.com

# Get the tokenizer.model from the github repo
cp "${CODE_DIR}/submodules/MGFM-train/train/minbpe/tokenizer/large-mgfm-1024.model" "${LOCAL_MODEL_DIR}/tokenizer.model"

echo "Wasabi checkpoint downloading process completes."
echo "The model checkpoint has been downloaded to ${LOCAL_MODEL_DIR}"
