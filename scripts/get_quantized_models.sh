#!/bin/bash

# Install the transformers package for quantization
pip install transformers
pip install optimum
pip install auto-gptq
pip install "bitsandbytes>=0.43.2"
pip install --upgrade torch torchvision torchaudio

ORIGINAL_MODEL_DIR=/workspace/MGFM/model_ckpts/step-00078000
ST_MODEL_DIR=/workspace/MGFM/model_ckpts/converted_safetensors/step-00078000
GPTQ_ST_CKPT_DIR="/workspace/MGFM/model_ckpts/converted_safetensors/gptq_safetensors/step-00078000"
NF4_ST_CKPT_DIR="/workspace/MGFM/model_ckpts/converted_safetensors/nf4_safetensors/step-00078000"

mkdir -p $GPTQ_ST_CKPT_DIR
mkdir -p $NF4_ST_CKPT_DIR

# Quantize safetensors using GPTQ
python MGFM-serving/src/quantize/quantize_safetensors.py --quant_type="gptq"
cp $ORIGINAL_MODEL_DIR/tokenizer.model $GPTQ_ST_CKPT_DIR/tokenizer.model
cp $ORIGINAL_MODEL_DIR/tokenizer.json $GPTQ_ST_CKPT_DIR/tokenizer.json

# Quantize safetensors using NF4
python MGFM-serving/src/quantize/quantize_safetensors.py --quant_type="nf4"
cp $ORIGINAL_MODEL_DIR/tokenizer.model $NF4_ST_CKPT_DIR/tokenizer.model
cp $ORIGINAL_MODEL_DIR/tokenizer.json $NF4_ST_CKPT_DIR/tokenizer.json
