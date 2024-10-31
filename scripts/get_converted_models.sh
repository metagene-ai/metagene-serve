#!/bin/bash

# Install litgpt package for sanity check and conversion
pip install 'litgpt[all]'

ORIGINAL_MODEL_DIR=/workspace/MGFM/model_ckpts/step-00078000
PTH_MODEL_DIR=/workspace/MGFM/model_ckpts/converted_pth/step-00078000
ST_MODEL_DIR=/workspace/MGFM/model_ckpts/converted_safetensors/step-00078000
# GGUF_MODEL_DIR=/workspace/MGFM/model_ckpts/converted_gguf/step-00078000

mkdir -p $PTH_MODEL_DIR
mkdir -p $ST_MODEL_DIR
# mkdir -p $GGUF_MODEL_DIR

# Get the tokenizer.model from the github repo
cp ./submodules/MGFM-training/train/minbpe/tokenizer/large-mgfm-1024.model $ORIGINAL_MODEL_DIR/tokenizer.model

# Convert litgpt format to pth format
litgpt convert_from_litgpt $ORIGINAL_MODEL_DIR $PTH_MODEL_DIR

cp $ORIGINAL_MODEL_DIR/tokenizer.model $PTH_MODEL_DIR/tokenizer.model
cp $ORIGINAL_MODEL_DIR/config.json $PTH_MODEL_DIR/config.json
cp $ORIGINAL_MODEL_DIR/tokenizer.json $PTH_MODEL_DIR/tokenizer.json

# Convert pth format to safetensors format
python ./src/convert/convert_pth_to_safetensors.py \
    --pth_model_dir=$PTH_MODEL_DIR \
    --st_model_dir=$ST_MODEL_DIR

cp $ORIGINAL_MODEL_DIR/tokenizer.model $ST_MODEL_DIR/tokenizer.model
cp $ORIGINAL_MODEL_DIR/config.json $ST_MODEL_DIR/config.json
cp $ORIGINAL_MODEL_DIR/tokenizer.json $ST_MODEL_DIR/tokenizer.json

# # Convert safetensors format to gguf format
# # https://www.substratus.ai/blog/converting-hf-model-gguf-model/
# gh repo clone ggerganov/llama.cpp

# conda config --add channels conda-forge
# conda install conda-forge::llama-cpp-python

# python /workspace/llama.cpp/convert_hf_to_gguf.py $ST_MODEL_DIR --outfile $GGUF_MODEL_DIR/model.gguf

# cp $ORIGINAL_MODEL_DIR/tokenizer.model $GGUF_MODEL_DIR/tokenizer.model
# cp $ORIGINAL_MODEL_DIR/config.json $GGUF_MODEL_DIR/config.json
# cp $ORIGINAL_MODEL_DIR/tokenizer.json $GGUF_MODEL_DIR/tokenizer.json
