#!/usr/bin/env python


import logging
from logging.config import dictConfig as _dictConfig
from os import path, environ

import yaml

__author__ = 'Samuel Marks'
__version__ = '0.0.63'


def get_logger(name=None):
    with open(path.join(path.dirname(__file__), '_data', 'logging.yml'), 'rt') as f:
        data = yaml.safe_load(f)
    _dictConfig(data)
    return logging.getLogger(name=name)


logger = get_logger('root')

environ.setdefault('TF', 'true')
environ.setdefault('TORCH', '')

if not ((environ['TF'] and 1 or 0) ^ (environ['TORCH'] and 1 or 0)):
    raise EnvironmentError('Only one of TensorFlow [`TF`] and PyTorch [`TORCH`] can be enabled')

if environ['TF']:
    import ml_glaucoma.tf_compat
    import ml_glaucoma.tfds_checksums

logging.getLogger('dataset_builder').setLevel(logging.WARNING)
