#!/bin/bash


RUN_DIR=$(dirname "$(readlink -f "$0")")
CORE_DIR=$(dirname "$(dirname "$RUN_DIR")")
SET_DIR="${CORE_DIR}/scripts/set"

MAMBA_ENV="metagene-evaluate"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
source "${SET_DIR}/env/set_vars.sh"


for dtype in \
    f16
do
    for file in \
        "tokenizer_rebuilt_prepend_eos.json"
    do
        filename="${file%.*}"
        output_dir="${OUTPUT_DIR}/evaluate/mteb/safetensors/${dtype}/${filename}"
        mkdir -p "${output_dir}"

        for ckpt in \
            step-00086000
        do
            echo "Copying ${file} to .../safetensors/${ckpt}/tokenizer.json ..."
            cp -f "./assets/tokenizer/${file}" "${MODEL_CKPT_DIR}/safetensors/${ckpt}/tokenizer.json"

            py_script="${CORE_DIR}/evaluate/evaluate_mteb.py"
            echo "Running ${py_script} with MGFM ${ckpt} in dtype ${dtype} ..."
            python "${py_script}" \
                --task_type HumanVirusReferenceClusteringP2P \
                --model_type "safetensors" \
                --model_dtype "${dtype}" \
                --model_dir "${MODEL_CKPT_DIR}" \
                --model_ckpt "${ckpt}" \
                --output_dir "${output_dir}" \
                --seed 42 \
#                --use_vllm
        done
    done
done

# for model_type in \
#    "DNABERT-2-117M" "DNABERT-S" \
#    "NT-v2-50m-multi-species" "NT-v2-100m-multi-species" "NT-v2-250m-multi-species" \
#    "NT-500m-1000g" "NT-500m-human-ref" "NT-v2-500m-multi-species" \
#    "NT-2.5b-multi-species" "NT-2.5b-1000g"
# do
#    py_script="${CORE_DIR}/evaluate/evaluate_mteb.py"
#    echo "Running ${py_script} with ${model_type} ..."
#    python "${py_script}" \
#        --task_type HumanVirusReferenceClusteringP2P \
#                    HumanVirusReferenceClusteringS2SAlign \
#                    HumanVirusReferenceClusteringS2SSmall \
#                    HumanVirusReferenceClusteringS2STiny \
#                    HumanMicrobiomeProjectReferenceClusteringP2P \
#                    HumanMicrobiomeProjectReferenceClusteringS2SAlign \
#                    HumanMicrobiomeProjectReferenceClusteringS2SSmall \
#                    HumanMicrobiomeProjectReferenceClusteringS2STiny \
#                    HumanMicrobiomeProjectReferenceClassificationMini \
#                    HumanMicrobiomeProjectDemonstrationMultiClassification \
#                    HumanMicrobiomeProjectDemonstrationClassificationDisease \
#                    HumanMicrobiomeProjectDemonstrationClassificationSex \
#                    HumanMicrobiomeProjectDemonstrationClassificationSource \
#                    HumanVirusClassificationOne \
#                    HumanVirusClassificationTwo \
#                    HumanVirusClassificationThree \
#                    HumanVirusClassificationFour \
#        --model_type "${model_type}" \
#        --model_dir "${MODEL_CKPT_DIR}" \
#        --output_dir "${OUTPUT_DIR}/evaluate/mteb"
# done