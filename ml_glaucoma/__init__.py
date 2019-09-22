#!/usr/bin/env python

import logging
from logging.config import dictConfig as _dictConfig
from os import path

import yaml

import ml_glaucoma.tf_compat
import ml_glaucoma.tfds_checksums

__author__ = 'Samuel Marks'
__version__ = '0.0.50'


def get_logger(name=None):
    with open(path.join(path.dirname(__file__), '_data', 'logging.yml'), 'rt') as f:
        data = yaml.safe_load(f)
    _dictConfig(data)
    return logging.getLogger(name=name)


logger = get_logger('root')
