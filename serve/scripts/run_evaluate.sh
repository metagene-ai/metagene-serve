#!/bin/bash


CONDA_ENV="metagene"
eval "$(conda shell hook --shell bash)" && conda activate "${CONDA_ENV}"


for model_name_or_path in \
    "metagene-ai/METAGENE-1" \
    "zhihan1996/DNABERT-2-117M" \
    "zhihan1996/DNABERT-S" \
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species" \
    "InstaDeepAI/NT-2.5b-1000g" \
    "InstaDeepAI/NT-500m-1000g" \
    "InstaDeepAI/NT-500m-human-ref" \
    "InstaDeepAI/NT-v2-100m-multi-species" \
    "InstaDeepAI/NT-v2-250m-multi-species" \
    "InstaDeepAI/NT-v2-500m-multi-species" \
    "InstaDeepAI/NT-v2-50m-multi-species"
do
    echo "Evaluating ${model_name_or_path} ..."
    py_script="${CORE_DIR}/evaluate/evaluate_mteb.py"
    python "${py_script}" \
        --task_name HumanVirusReferenceClusteringP2P \
                   HumanVirusReferenceClusteringS2SAlign \
                   HumanVirusReferenceClusteringS2SSmall \
                   HumanVirusReferenceClusteringS2STiny \
                   HumanMicrobiomeProjectReferenceClusteringP2P \
                   HumanMicrobiomeProjectReferenceClusteringS2SAlign \
                   HumanMicrobiomeProjectReferenceClusteringS2SSmall \
                   HumanMicrobiomeProjectReferenceClusteringS2STiny \
                   HumanMicrobiomeProjectReferenceClassificationMini \
                   HumanMicrobiomeProjectDemonstrationMultiClassification \
                   HumanMicrobiomeProjectDemonstrationClassificationDisease \
                   HumanMicrobiomeProjectDemonstrationClassificationSex \
                   HumanMicrobiomeProjectDemonstrationClassificationSource \
                   HumanVirusClassificationOne \
                   HumanVirusClassificationTwo \
                   HumanVirusClassificationThree \
                   HumanVirusClassificationFour \
        --model_name_or_path "${model_name_or_path}" \
        --seed 42
done