#!/bin/bash


RUN_DIR=$(dirname "$(readlink -f "$0")")
CORE_DIR=$(dirname "$(dirname "$RUN_DIR")")
SET_DIR="${CORE_DIR}/scripts/set"

source "${SET_DIR}/env/set_vars.sh"
MAMBA_ENV="mgfm_evaluate"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"

data_file="${DATA_DIR}/evaluate/sanity_check/cleaned_tokens_2000000000.txt"

# TODO add another file for vllm-based sanity check

for file in \
   tokenizer_rebuilt.json \
   tokenizer_rebuilt_prepend.json \
   tokenizer_rebuilt_bos.json \
   tokenizer_rebuilt_prepend_bos.json
do
    filename="${file%.*}"
    output_dir="${OUTPUT_DIR}/evaluate/sanity_check/${filename}"

    for ckpt in \
        step-00078000 \
        step-00080000 \
        step-00081000 \
        step-00082000 \
        step-00083000 \
        step-00084000 \
        step-00085000 \
        step-00086000
    do
        mkdir -p "${output_dir}/non_vllm/${ckpt}"
        echo "Creating output directory ${output_dir}/non_vllm/${ckpt} ..."

        echo "Copying ./test/tokenizer/${file} to /project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors/${ckpt}/tokenizer.json ..."
        cp -f "./test/tokenizer/${file}" /project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/safetensors/${ckpt}/tokenizer.json

        py_script="${CORE_DIR}/evaluate/evaluate_sanity_check.py"
        echo "Running ${py_script} with safetensors ckpt ${ckpt} ..."
        python "${py_script}" \
            --model_dir "${MODEL_CKPT_DIR}" \
            --model_type safetensors \
            --model_ckpt "${ckpt}" \
            --data_file "${data_file}" \
            --output_dir "${output_dir}" \
            --seed 42
    done
done