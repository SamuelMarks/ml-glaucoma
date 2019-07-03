#!/usr/bin/env bash

data_pardir="${DATA_DIR:-"$HOME"}"
manual_dir="${MANUAL_DIR:-/mnt}"

python -m ml_glaucoma v2 download \
	-ds bmes \
	--data_dir "$data_pardir"/tensorflow_datasets \
	--bmes_init \
	--manual_dir "$manual_dir" \
	--bmes_parent_dir "$data_pardir"/
