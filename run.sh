#!/usr/bin/env python

python -m ml_glaucoma pipeline \
          --options="{'losses': [{'BinaryCrossentropy': 0}, {'JaccardDistance': 0}],
                      'optimizer': {Adam: 5, RMSProp: 10}]}" \
          --strategy='grid|random|biggest-grid|smallest-grid|bayes|genetic|raytune' \
          train \
          -ds 'refuge' \
          --data_dir "$DOWNLOAD_DIR" \
          --model_file "$HOME"'/Projects/ml-glaucoma/ml_glaucoma/model_configs/applications.gin' \
          --model_dir "$HOME"'/Projects/gon_MobileNet_epochs_5' \
          --model_param "application = 'MobileNet'" \
          --epochs '1'

python -m ml_glaucoma pipeline \
  --options '{"models": [{"DenseNet169": 0}, {"EfficientNetB0": 0}], "losses": [{"BinaryCrossentropy": 0}, {"JaccardDistance": 0}], "optimizers": [{"Adadelta": 0}, {"Adagrad": 0}, {"Adam": 0}]}' \
  --key models --threshold 1 --logfile ./pipeline.log --dry-run train -ds refuge \
  --data_dir $HOME/mnt/tensorflow_datasets \
  --model_file $HOME/Projects/ml-glaucoma/ml_glaucoma/model_configs/applications.gin \
  --model_dir $HOME/Projects/gon_MobileNet_epochs_test --model_param application = "DenseNet169" \
  --epochs 250 --delete-lt 0.96 --losses BinaryCrossentropy --optimizers Adadelta --models DenseNet169
