from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

from ml_glaucoma.tfds_builders import refuge as _r


def RefugeConfig(resolution=(256, 256), rgb=True):
    if resolution is None:
        return _r.base_config
    return _r.RefugeConfig(resolution=resolution, rgb=rgb)


def Refuge(config=None, task=_r.RefugeTask.CLASSIFICATION, data_dir=None):
    if config is None:
        config = RefugeConfig()
    return _r.Refuge(config=config, task=task, data_dir=data_dir)
