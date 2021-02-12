#!/bin/sh

# This scripts is a helper to split the dataset chronologically.
# We keep the temporal order of the dossiers.
# In this case, we filter the dataset first, because removing some
# dossiers with few edits will change the split indices.
#
# It requires the following arguments:
# - $1: path to folder of "canonical" dataset
# - $2: output path

python split-chronologically.py \
    --canonical $1 \
    --output_path $2 \
    --order chronological \
    --threshold 10 \
    --split_train 0.8 \
    --split_valid 0.9
