#!/bin/bash


# Common dependencies
source ./scripts/bash/set/set_env_basic.sh

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

pip install accelerate
pip install deepspeed
