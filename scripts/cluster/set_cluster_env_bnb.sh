#!/bin/bash


# install cmake without sudo
# cd ~
# wget https://github.com/Kitware/CMake/releases/download/v3.31.0/cmake-3.31.0.tar.gz
# tar -zxvf cmake-3.31.0.tar.gz && cd cmake-3.31.0
#
# ./bootstrap --prefix=$HOME/cmake
# make
# make install
# export PATH=$HOME/cmake/bin:$PATH
# source ~/.bashrc
# cmake --version
pip install cmake

# override pytorch cuda
# https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend
cd ~
mkdir -p ~/local
wget https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/install_cuda.sh
bash install_cuda.sh 118 ~/local 1

# Add the following to .bashrc
echo "export BNB_CUDA_VERSION=118" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$HOME/local/cuda-11.8" >> ~/.bashrc

# build bitsandbytes from source
cd ~
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && cd bitsandbytes/
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cuda -S . && make
pip install -e .
