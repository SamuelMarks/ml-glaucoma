"""
Module for creating `tensorflow_datasets` `DatasetBuilder`s.

https://github.com/tensorflow/datasets/blob/master/docs/add_dataset.md

This module should be kept independent of any model-specific preprocessing
or model implementations. See `ml_glaucoma.problems` for model-specific adapters
(data augmentation, model-specific data pipeline etc.).
"""

from os import environ

if not environ['TF']:
    raise NotImplementedError('tensorflow_datasets is [currently] TensorFlow only')
