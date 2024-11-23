#!/bin/bash


## comment the following if in a slurm-based env
#source ./scripts/bash/set/set_env_vars.sh

pytest -s "${CODE_DIR}/src/test/test_env.py"
