#!/bin/bash


pip install gdown
gdown_path="1hbq0BTS0zbVS8Y708NE4_O21TmRuIM8B"
SANITY_CHECK_DATA_DIR="../data/sanity_check"

mkdir -p $SANITY_CHECK_DATA_DIR
gdown $gdown_path -O $SANITY_CHECK_DATA_DIR
