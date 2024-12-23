#!/bin/bash


RUN_DIR=$(dirname "$(readlink -f "$0")")
CORE_DIR=$(dirname "$(dirname "$RUN_DIR")")
SET_DIR="${CORE_DIR}/scripts/set"

source "${SET_DIR}/env/set_vars.sh"
MAMBA_ENV="metagene_gue"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"


timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
run_name="gue_hpo_${timestamp}"
output_dir="${OUTPUT_DIR}/optuna/${run_name}"
logging_dir="${LOGGING_DIR}/optuna/${run_name}"
mkdir -p "${output_dir}"
mkdir -p "${logging_dir}"

start_time=$(date +%s)

model_name_or_path="/expanse/lustre/projects/mia346/swang31/projects/MGFM/MGFM-serving/model_ckpts/safetensors/step-00086000"
python "./serving/finetune/hpo/main.py" \
    --seed 42 \
    --n_trials 16 \
    --model_name_or_path "${model_name_or_path}" \
    --no-use_4bit_quantization \
    --data_dir "./assets/data/GUE/EMP/H3" \
    --accelerate_config "./serving/finetune/hpo/config/accel_config_ds_z3.yaml" \
    --run_name "${run_name}" \
    --output_dir "${output_dir}" \
    --logging_dir "${logging_dir}" \
    --wandb_project "metagene_evaluation" \
    --wandb_name "${run_name}" \
    --wandb_api_key "${WANDB_API_KEY}" \
    --hf_token "${HF_TOKEN}" \
#    --wandb_run_id "${wandb_run_id}"

end_time=$(date +%s)
tot_time=$((end_time - start_time))
tot_time=$((tot_time / 60))
echo "Elapsed time: ${tot_time} mins"

echo "Check ${logging_dir} for other logs"
echo "Check ${output_dir} for the output"