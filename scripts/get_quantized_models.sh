#!/bin/bash

ORIGINAL_MODEL_DIR="../model_ckpts/litgpt/step-00078000"
ST_MODEL_DIR="../model_ckpts/safetensors/step-00078000"
NF4_ST_CKPT_DIR="../model_ckpts/safetensors/nf4/step-00078000"
# GPTQ_ST_CKPT_DIR="../model_ckpts/safetensors/gptq/step-00078000"

mkdir -p $NF4_ST_CKPT_DIR
# mkdir -p $GPTQ_ST_CKPT_DIR

# Quantize safetensors using NF4
python ./src/quantize/quantize_st.py --quant_type="nf4"
cp $ORIGINAL_MODEL_DIR/tokenizer.model $NF4_ST_CKPT_DIR/tokenizer.model
cp $ORIGINAL_MODEL_DIR/tokenizer.json $NF4_ST_CKPT_DIR/tokenizer.json

## Quantize safetensors using GPTQ
#python ./src/quantize/quantize_st.py --quant_type="gptq"
#cp $ORIGINAL_MODEL_DIR/tokenizer.model $GPTQ_ST_CKPT_DIR/tokenizer.model
#cp $ORIGINAL_MODEL_DIR/tokenizer.json $GPTQ_ST_CKPT_DIR/tokenizer.json