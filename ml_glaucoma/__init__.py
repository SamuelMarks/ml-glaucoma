#!/usr/bin/env python

import logging
from logging.config import dictConfig as _dictConfig
from os import path

import yaml

__author__ = 'Samuel Marks'
__version__ = '0.0.12'


def get_logger(name=None):
    with open(path.join(path.dirname(__file__), '_data', 'logging.yml'), 'rt') as f:
        data = yaml.load(f)
    _dictConfig(data)
    return logging.getLogger(name=name)


logger = get_logger('root')
