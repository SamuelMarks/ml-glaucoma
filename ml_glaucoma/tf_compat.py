"""Monkey-patches tensorflow versions to be 2.0 compatible."""

from __future__ import division
from __future__ import print_function

from os import environ

from tensorflow.python.ops.losses.loss_reduction import ReductionV2

if not environ['TF']:
    raise NotImplementedError('tf_compat is TensorFlow only')

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        'No tensorflow installation found. `ml-glaucoma` does not '
        'automatically install tensorflow. Please install manually.')
import distutils.version

tf_version = distutils.version.LooseVersion(tf.__version__)
is_tf_v1 = tf_version.version[0] == 1
is_tf_v2 = tf_version.version[0] == 2

if not (is_tf_v1 or is_tf_v2):
    raise ImportError(
        'Detected version of tensorflow {:s} not compatible with `ml-glaucoma` -'
        ' only versions 1 and 2 supported'.format(tf.__version__))

if is_tf_v1:
    def dim_value(dimension):
        return dimension.value


    if tf_version.version[1] == 13:
        from tensorflow.keras.losses import Loss
        from tensorflow.python.ops.losses import losses_impl

        tf.keras.losses.Loss = Loss
        tf.keras.losses.Reduction = ReductionV2
        del losses_impl
        del Loss
    elif tf_version.version[1] == 14:
        from tensorflow.keras.utils import losses_utils

        tf.keras.losses.Reduction = losses_utils.ReductionV2
        del losses_utils
else:
    def dim_value(dimension):
        return dimension

if is_tf_v1:
    tf.nest = tf.contrib.framework.nest
    from tensorflow.keras.metrics import Metric

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
