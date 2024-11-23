#!/bin/bash


# run with debug config
CUDA_LAUNCH_BLOCKING=1 python "${CODE_DIR}/src/evaluation/eval_GUE.py"
