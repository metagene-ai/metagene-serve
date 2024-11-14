#!/bin/bash


echo "Downloading HF model checkpoint from Wasabi ..."

# Get Wasabi path-to-model
REMOTE_MODEL_DIR="s3://mgfm-02/model-weights/hf-ckpt-78k"
LOCAL_MODEL_DIR="../model_ckpts/safetensors/step-00078000"
mkdir -p $LOCAL_MODEL_DIR

aws s3 sync $REMOTE_MODEL_DIR $LOCAL_MODEL_DIR --endpoint-url=https://s3.us-west-1.wasabisys.com

echo "Wasabi checkpoint downloading procoesss completes."
echo "The model checkpoint has been downloaded to ~/workspace/MGFM/model_ckpts/safetensors/step-00078000"


echo "Downloading data for fine-tuning ..."
GUE_PATH="1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2"
GUE_DIR="../data/fine-tune/"

mkdir -p $GUE_DIR
pip install gdown
gdown $GUE_PATH -O $GUE_DIR
unzip -q "$GUE_DIR/GUE.zip" -d $GUE_DIR && rm "$GUE_DIR/GUE.zip"
