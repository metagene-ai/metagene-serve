#!/bin/bash


cd ~
wget https://github.com/Kitware/CMake/releases/download/v3.31.2/cmake-3.31.2.tar.gz
tar -zxvf cmake-3.31.2.tar.gz
cd cmake-3.31.2
./bootstrap --prefix="${HOME}/local"
make && make install

# Add cmake to PATH
#export PATH=$HOME/local/bin:$PATH