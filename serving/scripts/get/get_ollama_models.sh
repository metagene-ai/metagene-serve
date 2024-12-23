#!/bin/bash


RUN_DIR=$(dirname "$(readlink -f "$0")")
CORE_DIR=$(dirname "$(dirname "$RUN_DIR")")
SET_DIR="${CORE_DIR}/scripts/set"

source "${SET_DIR}/env/set_vars.sh"
MAMBA_ENV="metagene-1"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"

#     tokenizer_rebuilt.json \
#     tokenizer_rebuilt_bos.json \
#     tokenizer_rebuilt_eos.json \
#     tokenizer_rebuilt_bos_eos.json \
#     tokenizer_rebuilt_prepend.json \
#     tokenizer_rebuilt_prepend_bos.json \
#     tokenizer_rebuilt_prepend_eos.json \
#     tokenizer_rebuilt_prepend_bos_eos.json

# tokenizer_rebuilt.json \
for file in \
    tokenizer_rebuilt_prepend.json
do
    filename="${file%.*}"
    echo "Copying ${file} to .../safetensors/step-00086000/tokenizer.json ..."
    cp -f "./test/tokenizer/${file}" "${MODEL_CKPT_DIR}/safetensors/step-00086000/tokenizer.json"

    ollama create "st_86k_${filename}" -f ./serving/inference/ollama/metagene_st/Modelfile
done