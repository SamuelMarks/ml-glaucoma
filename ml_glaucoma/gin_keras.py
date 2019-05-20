from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from gin import config


def _register_callables(package, module, blacklist):
  for k in dir(package):
    if k not in blacklist:
      v = getattr(package, k)
      if callable(v):
        config.external_configurable(v, name=k, module=module)


blacklist = set(('serialize', 'deserialize', 'get'))
# These may end up moving into gin-config proper
for package, module in (
    (tf.keras.losses, 'tf.keras.losses'),
    (tf.keras.metrics, 'tf.keras.metrics'),
    (tf.keras.optimizers, 'tf.keras.optimizers'),
    ):
  _register_callables(package, module, blacklist)


# clean up namespace
del package, module, blacklist
