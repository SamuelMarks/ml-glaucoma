#!/usr/bin/env bash

dc_gin="$(python -c 'from pkg_resources import resource_filename; from os import path; print(path.join(path.dirname(resource_filename("ml_glaucoma", "__init__.py")), "model_configs", "dc.gin"))')"
data_pardir="${DATA_DIR:-"$HOME"}"
manual_dir="${MANUAL_DIR:-/mnt}"

python -m ml_glaucoma train \
    -ds bmes \
    --data_dir "$data_pardir"/tensorflow_datasets \
    --model_file "$dc_gin" \
    --model_dir "$data_pardir"/ml_glaucoma_models/bmes-n0
