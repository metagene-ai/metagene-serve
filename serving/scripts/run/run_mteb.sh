#!/bin/bash


RUN_DIR=$(dirname "$(readlink -f "$0")")
CORE_DIR=$(dirname "$(dirname "$RUN_DIR")")
SET_DIR="${CORE_DIR}/scripts/set"

source "${SET_DIR}/env/set_vars.sh"
MAMBA_ENV="mgfm_evaluate"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"


for file in \
    tokenizer_rebuilt_prepend_bos.json \
    tokenizer_rebuilt.json \
    tokenizer_rebuilt_prepend.json \
    tokenizer_rebuilt_bos.json
do
    filename="${file%.*}"
    output_dir="${OUTPUT_DIR}/evaluate/mteb/safetensors/${filename}"
    mkdir -p "${output_dir}"

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
        echo "Copying ${file} to .../safetensors/${ckpt}/tokenizer.json ..."
        cp -f "./test/tokenizer/${file}" "${MODEL_CKPT_DIR}/safetensors/${ckpt}/tokenizer.json"

        py_script="${CORE_DIR}/evaluate/evaluate_mteb_vllm.py"
        echo "Running ${py_script} with safetensors ckpt ${ckpt} and ${filename} ..."
        python "${py_script}" \
            --task_type HumanMicrobiomeProjectClassificationTest \
            --model_ckpt ${ckpt} \
            --output_dir "${output_dir}"

        py_script="${CORE_DIR}/evaluate/evaluate_mteb.py"
        echo "Running ${py_script} with MGFM ${ckpt} ..."
        python "${py_script}" \
            --task_type HumanMicrobiomeProjectClassificationTest \
            --model_type "MGFM" \
            --model_dir "${MODEL_CKPT_DIR}" \
            --model_ckpt "${ckpt}" \
            --output_dir "${output_dir}"
    done
done


for model_type in \
    "DNABERT-2-117M" "DNABERT-S" \
    "NT-v2-50m-multi-species" "NT-v2-100m-multi-species" "NT-v2-250m-multi-species" \
    "NT-500m-1000g" "NT-500m-human-ref" "NT-v2-500m-multi-species" \
    "NT-2.5b-multi-species" "NT-2.5b-1000g"
do
    py_script="${CORE_DIR}/evaluate/evaluate_mteb.py"
    echo "Running ${py_script} with ${model_type} ..."
    python "${py_script}" \
        --task_type HumanMicrobiomeProjectClassificationTest \
        --model_type "${model_type}" \
        --model_dir "${MODEL_CKPT_DIR}" \
        --output_dir "${OUTPUT_DIR}/evaluate/mteb"
done

#         --task_type HumanMicrobiomeProjectClassificationTest \
#                     HumanVirusClassificationOneTest \
#                     HumanVirusClassificationTwoTest \
#                     HumanVirusClassificationThreeTest \
                    HumanVirusClassificationFourTest \