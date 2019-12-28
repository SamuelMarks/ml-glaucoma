#!/usr/bin/env bash

data_dir="${DATA_DIR:-"$HOME"/tensorflow_datasets}"
parent_dir="${DATASET_PARENT_DIR:-"$HOME"}"
manual_dir="${MANUAL_DIR:-/mnt}"
dataset="${DATASET:-dr_spoc}"

python -m ml_glaucoma download \
    -ds "$dataset" \
    --data_dir "$data_dir" \
    --"$dataset"_init \
    --manual_dir "$manual_dir" \
    --"$dataset"_parent_dir "$parent_dir"/
