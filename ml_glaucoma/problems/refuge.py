from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

from ml_glaucoma.tfds_builders import refuge as _r


@gin.configurable
def RefugeConfig(resolution=(256, 256)):
    if resolution is None:
        return _r.base_config
    return _r.RefugeConfig(resolution=resolution)


@gin.configurable
def Refuge(config=None, task=_r.RefugeTask.CLASSIFICATION):
    if config is None:
        config = RefugeConfig()
    return _r.Refuge(config=config, task=task)
