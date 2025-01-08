#!/bin/bash


RUN_DIR=$(dirname "$(readlink -f "$0")")
CORE_DIR=$(dirname "$(dirname "$RUN_DIR")")
SET_DIR="${CORE_DIR}/scripts/set"

MAMBA_ENV="metagene-evaluate"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
source "${SET_DIR}/env/set_vars.sh"

data_file="${DATA_DIR}/evaluate/sanity_check/cleaned_tokens_2000000000.txt"


for dtype in f16
do
    for file in tokenizer_rebuilt_prepend.json
    do
        filename="${file%.*}"
        output_dir="${OUTPUT_DIR}/evaluate/sanity_check/safetensors/${filename}/non_vllm/${dtype}"

        if [[ "${filename}" == "tokenizer_rebuilt" || "${filename}" == "tokenizer_rebuilt_bos" ]];
        then
            use_prepend=1
        else
            use_prepend=0
        fi

        for ckpt in step-00086000
        do
            mkdir -p "${output_dir}/${ckpt}"
            echo "Creating output directory ${output_dir}/${ckpt} ..."

            if [[ "${dtype}" == "bnb-4bit" || "${dtype}" == "bnb-8bit" || "${dtype}" == "gptq-4bit" || "${dtype}" == "gptq-8bit" || "${dtype}" == "awq-4bit" || "${dtype}" == "quanto-4bit" ]];
            then
                use_quantized=1
                echo "Copying ./assets/tokenizer/${file} to ... /MGFM/MGFM-serving/model_ckpts/safetensors/${dtype}/${ckpt}/tokenizer.json ..."
                cp -f "./assets/tokenizer/${file}" "${MODEL_CKPT_DIR}/safetensors/${dtype}/${ckpt}/tokenizer.json"
            else
                use_quantized=0
                echo "Copying ./assets/tokenizer/${file} to ... /MGFM/MGFM-serving/model_ckpts/safetensors/${ckpt}/tokenizer.json ..."
                cp -f "./assets/tokenizer/${file}" "${MODEL_CKPT_DIR}/safetensors/${ckpt}/tokenizer.json"
            fi

            py_script="${CORE_DIR}/evaluate/evaluate_sanity_check.py"
            echo "Running ${py_script} with safetensors ckpt ${ckpt} in dtype ${dtype} ..."
            python "${py_script}" \
                --model_dir "${MODEL_CKPT_DIR}" \
                --model_type safetensors \
                --model_dtype "${dtype}" \
                --model_ckpt "${ckpt}" \
                --data_file "${data_file}" \
                --output_dir "${output_dir}" \
                --use_vllm "${use_vllm}" \
                --use_prepend "${use_prepend}" \
                --use_quantized "${use_quantized}" \
                --seed 42
        done
    done
done

#for file in \
#    tokenizer_rebuilt_bos_no_eos.json
#do
#    filename="${file%.*}"
#    output_dir="${OUTPUT_DIR}/evaluate/sanity_check/litgpt/${filename}"
#
#    if [[ "${filename}" == "tokenizer_rebuilt" || "${filename}" == "tokenizer_rebuilt_bos" ]];
#    then
#        use_prepend=1
#    else
#        use_prepend=0
#    fi
#
#    for ckpt in \
#        step-00078000 \
#        step-00086000
#    do
#        echo "Copying ${file} to .../litgpt/${ckpt}/tokenizer.json ..."
#        cp -f "./test/tokenizer/${file}" /project/neiswang_1391/MGFM/MGFM-serving/model_ckpts/litgpt/${ckpt}/tokenizer.json
#        mkdir -p "${output_dir}/${ckpt}"
#
#        py_script="${CORE_DIR}/evaluate/evaluate_sanity_check_litgpt.py"
#        echo "Running sanity check with litgpt ${ckpt} and ${file} ..."
#        python "${py_script}" \
#            --model_dir "${MODEL_CKPT_DIR}" \
#            --model_type litgpt \
#            --model_ckpt "${ckpt}" \
#            --data_file "${data_file}" \
#            --output_dir "${output_dir}" \
#            --use_prepend "${use_prepend}" \
#            --seed 42
#    done
#done