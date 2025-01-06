#!/bin/bash


RUN_DIR=$(dirname "$(readlink -f "$0")")
CORE_DIR=$(dirname "$(dirname "$RUN_DIR")")
SET_DIR="${CORE_DIR}/scripts/set"

MAMBA_ENV="metagene-1-hpo"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
source "${SET_DIR}/env/set_vars.sh"



timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
run_name="gue_hpo_${timestamp}"
wandb_run_id=""
output_dir="${OUTPUT_DIR}/optuna/${run_name}"
logging_dir="${LOGGING_DIR}/optuna/${run_name}"
if [ "${wandb_run_id}" == "" ]; then
    rm -rf "${output_dir}" && mkdir -p "${output_dir}"
    rm -rf "${logging_dir}" && mkdir -p "${logging_dir}"
fi

start_time=$(date +%s)

model_name_or_path="${MODEL_CKPT_DIR}/safetensors/step-00086000"
data_dir="./assets/data/GUE"



#mouse/0 30 (full)
# * 32bs 1epoch: 0.601
# * 8bs 1epoch: 0.621

#mouse/1 (full)
# * 8bs 2epoch: running

#mouse/2 (full)
# * 8bs 1epoch: 0.830

#mouse/3 (full)
# * 8bs 2epoch: 0.825

#mouse/4 (full)
# * 8bs 3epoch: running

#tf/0 (q,k,v) 30 (full?)
# * 32bs 1epoch: 0.687
# * 8bs 3epoch: running

#prom/prom_300_all 70 (full)
# * 32bs 1epoch: 0.867
# * 8bs 3epoch: running

#prom/prom_core_all 20 (full)
# * 32bs 1epoch: 0.661
# * 8bs 3epoch: running

#splice/reconstructed (full)
# * 32bs 1epoch: 0.829
# * 8bs 2epoch: running

python "./serving/finetune/hpo/main.py" \
    --seed 42 \
    --n_trials 1 \
    --model_name_or_path "${model_name_or_path}" \
    --no-use_4bit_quantization \
    --data_dir "${data_dir}/splice/reconstructed" \
    --model_max_length 80 \
    --num_train_epochs 2 \
    --accelerate_config "./serving/finetune/hpo/config/accel_config_ds_z3.yaml" \
    --run_name "${run_name}" \
    --output_dir "${output_dir}" \
    --logging_dir "${logging_dir}" \
    --wandb_project "metagene_evaluation" \
    --wandb_name "${run_name}" \
    --wandb_api_key "${WANDB_API_KEY}" \
    --hf_token "${HF_TOKEN}" \
    --wandb_run_id "${wandb_run_id}"

end_time=$(date +%s)
tot_time=$((end_time - start_time))
tot_time=$((tot_time / 60))
echo "Elapsed time: ${tot_time} mins"

echo "Check ${logging_dir} for other logs"
echo "Check ${output_dir} for the output"