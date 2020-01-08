#!/usr/bin/env bash

data_dir="${DATA_DIR:-"$HOME"/tensorflow_datasets}"
transfer_gin="$(python -c 'from pkg_resources import resource_filename; from os import path; print(path.join(path.dirname(resource_filename("ml_glaucoma", "__init__.py")), "model_configs", "transfer.gin"))')"
manual_dir="${MANUAL_DIR:-/mnt}"
dataset="${DATASET:-dr_spoc}"

python -m ml_glaucoma pipeline \
    --options="{ losses: [{BinaryCrossentropy: 0}, {JaccardDistance: 0}],
                 optimizer: [{Adam: 5}, {RMSProp: 10}] }" \
    --strategy='grid|random|biggest-grid|smallest-grid|bayes|genetic|raytune' \
    train \
    -ds "$dataset" \
    --data_dir "$data_dir" \
    --model_file "$transfer_gin" \
    --model_dir "$manual_dir"'/gon_MobileNet_epochs_5' \
    --model_param 'application = "MobileNet"' \
    --epochs '1'

python -m ml_glaucoma pipeline \
    --options="{ models: [{DenseNet169: 0}, {EfficientNetB0: 0}],
                 losses: [{BinaryCrossentropy: 0}, {JaccardDistance: 0}],
                 optimizers: [{Adadelta: 0}, {Adagrad: 0}, {Adam: 0}] }" \
    --key 'models' \
    --threshold '1' \
    --logfile "$manual_dir"'/pipeline.log' \
    --dry-run 'train' \
    -ds "$dataset" \
    --data_dir "$data_dir" \
    --model_file "$transfer_gin" \
    --model_dir "$manual_dir"'/gon_MobileNet_epochs_test' \
    --model_param 'application = "DenseNet169"' \
    --epochs '250' \
    --delete-lt '0.96' \
    --losses 'BinaryCrossentropy' \
    --optimizers 'Adadelta' \
    --models 'DenseNet169'
