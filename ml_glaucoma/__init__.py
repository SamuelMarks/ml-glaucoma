#!/usr/bin/env python

import logging.config
from os import path, environ

__author__ = 'Samuel Marks'
__version__ = '0.0.76'


def get_logger(name=None):
    logging.config.fileConfig(path.join(path.dirname(__file__), '_data', 'logging.conf'))
    return logging.getLogger(name=name)


logging.getLogger('matplotlib').disabled = True
logger = get_logger('root')

environ.setdefault('TF', 'true')
environ.setdefault('TORCH', '')

if not ((environ['TF'] and 1 or 0) ^ (environ['TORCH'] and 1 or 0)):
    raise EnvironmentError('Only one of TensorFlow [`TF`] and PyTorch [`TORCH`] can be enabled')

if environ['TF']:
    import ml_glaucoma.tf_compat
    import ml_glaucoma.tfds_checksums

# logging.getLogger('dataset_builder').setLevel(logging.WARNING)
# logging.getLogger('matplotlib').setLevel(logging.ERROR)
