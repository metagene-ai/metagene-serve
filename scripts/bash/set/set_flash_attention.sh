#!/bin/bash


echo "Configuring flash attention ..."
cd ~
gh repo clone triton-lang/triton
cd triton/python
pip install cmake
pip install -e .