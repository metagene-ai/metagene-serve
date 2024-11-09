# MGFM

## Quick Tour

* [`get_sample_data.sh`](get_sample_data.sh): get sample data from remote storage for sanity checks.
* [`get_finetune_data.sh`](get_finetune_data.sh): get the data from the GUE benchmark for fine-tuning.
* [`get_converted_models.sh`](get_converted_models.sh): set up the env for model format conversion and perform conversion.
* [`get_quantized_models.sh`](get_quantized_models.sh): set up the env for [GPTQ](https://arxiv.org/abs/2210.17323) and [NF4](https://huggingface.co/blog/4bit-transformers-bitsandbytes) quantization and perform quantization.
* [`run_sanity_check.sh`](../run/run_sanity_check.sh): pull dataset for sanity check and perform the sanity check on given models.
* [`slurm_benchmark.sh`](slurm_benchmark.sh): fine-tune and evaluate the model on the GUE benchmark.
