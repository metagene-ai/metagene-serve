#!/bin/bash
model_path="/workspace/MGFM/model_ckpts/converted_safetensors/step-00078000"
data_path="/workspace/MGFM/data/fine-tune/GUE"
output_dir="/workspace/MGFM/model_ckpts/fine-tune/step-00078000"

lr=3e-5
seed=42

gpu_list=$(nvidia-smi -L)
gpu_count=$(echo "$gpu_list" | grep -c "GPU")

echo "Fine-tuning the model for the GUE benchmark ..."

echo "Fine-tuning the model on EMP ..."
for data in H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac
do
    if [ "$gpu_count" -gt 1 ]; then
        echo "Fine-tuning the model on GUE/EMP/$data with $gpu_count GPUs"
        torchrun --nproc_per_node $gpu_count ./src/finetune/finetune_full.py \
            --model_name_or_path $model_path \
            --output_dir $output_dir \
            --data_path  $data_path/EMP/$data \
            --run_name MGFM_${lr}_EMP_${data}_seed${seed} \
            --model_max_length 128 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 3 \
            --save_steps 200 \
            --eval_steps 200 \
            --warmup_steps 50
    else
        echo "Fine-tuning the model on GUE/EMP/$data with a single GPU"
        python ./src/finetune/finetune_full.py \
            --model_name_or_path $model_path \
            --output_dir $output_dir \
            --data_path  $data_path/EMP/$data \
            --run_name MGFM_${lr}_EMP_${data}_seed${seed} \
            --model_max_length 128 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 3 \
            --save_steps 200 \
            --eval_steps 200 \
            --warmup_steps 50
        fi
done

echo "Finish Fine-tuning the model"