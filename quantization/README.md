# Quantization

This directory contains files for model quantization for low-resource deployment.

## Quick Tour
* [quantize_safetensors.py](quantize_safetensors.py): perform 4-bit quantization for model in safetensors format using [GPTQ](https://arxiv.org/abs/2210.17323) and [NF4](https://huggingface.co/blog/4bit-transformers-bitsandbytes).
