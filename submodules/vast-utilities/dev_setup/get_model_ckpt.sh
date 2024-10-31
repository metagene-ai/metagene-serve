#!/bin/bash

echo "Downloading model checkpoint from Wasabi ..."

# Get Wasabi path-to-model 
get_input() {
    read -p "$1: " value
    echo $value
}
REMOTE_MODEL_DIR=$(get_input "Enter your remote Wasabi model dir path")

# Get local model ckpt dir 
get_input() {
    read -p "$1: " value
    echo $value
}
LOCAL_MODEL_DIR=$(get_input "Enter your local model dir path")
mkdir -p $LOCAL_MODEL_DIR

aws s3 sync $REMOTE_MODEL_DIR $LOCAL_MODEL_DIR --endpoint-url=https://s3.us-west-1.wasabisys.com

echo "Wasabi checkpoint downloading procoesss completes."
echo "The model checkpoint has been downloaded to model_ckpts/"
