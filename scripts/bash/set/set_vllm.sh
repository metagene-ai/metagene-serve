#!/bin/bash


pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

# # run the following to get into the dev mode
# cd ./src/finetune/vllm
# python python_only_dev.py

# # run the following before git push
# python python_only_dev.py --quit-dev
