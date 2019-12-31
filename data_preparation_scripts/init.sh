#!/usr/bin/env bash

data_dir="${DATA_DIR:-"$HOME"/tensorflow_datasets}"
parent_dir="${DATASET_PARENT_DIR:-"$HOME"}"
manual_dir="${MANUAL_DIR:-/mnt}"
dataset="${DATASET:-dr_spoc}"
if [ "${dataset#dr_spoc}" != "${dataset}" ]; then
    prefix='dr_spoc';
else
    prefix="$dataset";
fi

python -m ml_glaucoma download \
    -ds "$dataset" \
    --data_dir "$data_dir" \
    --"$prefix"_init \
    --manual_dir "$manual_dir" \
    --"$prefix"_parent_dir "$parent_dir"/
