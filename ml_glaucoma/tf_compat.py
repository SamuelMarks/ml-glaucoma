"""Monkey-patches tensorflow versions to be 2.0 compatible."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        'No tensorflow installation found. `gl-glaucoma` does not '
        'automatically install tensorflow. Please install manaully.')
import distutils.version
tf_version = distutils.version.LooseVersion(tf.__version__)
is_v1 = tf_version.version[0] == 1
is_v2 = tf_version.version[0] == 2

if not (is_v1 or is_v2):
    raise ImportError(
        'Detected version of tensorflow %s not compatible with `gl-glaucoma` -'
        ' only versions 1 and 2 supported' % (tf.__version__))

if is_v1:
    def dim_value(dimension):
        return dimension.value
    from tensorflow.python.keras.losses import Loss
    from tensorflow.python.ops.losses import losses_impl
    tf.keras.losses.Loss = Loss
    tf.keras.losses.Reduction = losses_impl.ReductionV2
    del Loss
    del losses_impl
else:
    def dim_value(dimension):
        return dimension


if is_v1:
    tf.nest = tf.contrib.framework.nest
    from tensorflow.python.keras.metrics import Metric
    tf.keras.metrics.Metric = Metric
    del Metric
    if not hasattr(tf.keras.losses, 'SparseCategoricalCrossentropy'):
        tf.keras.losses.SparseCategoricalCrossentropy = \
            tf.keras.losses.CategoricalCrossentropy

    if not hasattr(tf, 'compat'):
        class _Compat(object):
            v1 = tf

        tf.compat = _Compat
        del _Compat
