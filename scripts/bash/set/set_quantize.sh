#!/bin/bash


echo "Downloading HF model checkpoint from Wasabi ..."

# Get Wasabi path-to-model
REMOTE_MODEL_DIR="s3://mgfm-02/model-weights/hf-ckpt-78k"
LOCAL_MODEL_DIR="../model_ckpts/safetensors/step-00078000"
mkdir -p $LOCAL_MODEL_DIR

aws s3 sync $REMOTE_MODEL_DIR $LOCAL_MODEL_DIR --endpoint-url=https://s3.us-west-1.wasabisys.com

echo "Wasabi checkpoint downloading procoesss completes."
echo "The model checkpoint has been downloaded to ~/workspace/MGFM/model_ckpts/safetensors/step-00078000"


pip install gdown
gdown_path="1hbq0BTS0zbVS8Y708NE4_O21TmRuIM8B"
SANITY_CHECK_DATA_DIR="../data/sanity_check"

mkdir -p $SANITY_CHECK_DATA_DIR
gdown $gdown_path -O $SANITY_CHECK_DATA_DIR

