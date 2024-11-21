#!/bin/bash


source ./scripts/bash/set/set_env_basic.sh

# for sanity check and quantization
GDOWN_PATH="1hbq0BTS0zbVS8Y708NE4_O21TmRuIM8B" # cleaned_tokens_2000000000.txt
DATA_TYPE="sanity_check"

## for GUE benchmark
#GDOWN_PATH="1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2" # GUE.zip
#DATA_TYPE="evaluation"

LOCAL_DATA_DIR="${DATA_DIR}/${DATA_TYPE}"
mkdir -p "${LOCAL_DATA_DIR}"
gdown "${GDOWN_PATH}" -O "${LOCAL_DATA_DIR}/cleaned_tokens_2000000000.txt"
#gdown "${GDOWN_PATH}" -O "${LOCAL_DATA_DIR}/GUE.zip"

#unzip -q "$GUE_DIR/GUE.zip" -d $GUE_DIR && rm "$GUE_DIR/GUE.zip"
