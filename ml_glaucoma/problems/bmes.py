from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

from ml_glaucoma.tfds_builders import bmes as _b


@gin.configurable
def BmesConfig(resolution=(256, 256)):
    if resolution is None:
        return _b.Bmes.base_config
    return _r.BmesConfig(resolution=resolution)


@gin.configurable
def Bmes(config=None):
    if config is None:
        config = RefugeConfig()
    return _b.Bmes(config=config)
