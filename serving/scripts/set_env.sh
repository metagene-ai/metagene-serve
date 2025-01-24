#!/bin/bash
# python 3.10 + cuda 11.8.0


conda clean -a -y

# create a new conda environment
conda create -n metagene python=3.10 -y

# cuda, gcc/g++, and torch
conda install cuda -c nvidia/label/cuda-11.8.0 -y
conda install pytorch=2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# bitsandbytes
conda install bitsandbytes=0.45.0 -c conda-forge --no-pyc -y

# autoawq
pip install autoawq

# quanto
pip install optimum-quanto

# gene-mteb
MTEB_PATH="./gene-mteb"
git clone https://github.com/metagene-ai/gene-mteb.git "${MTEB_PATH}"
cd "${MTEB_PATH}" && pip install -e . && cd -