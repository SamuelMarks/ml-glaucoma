from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin
from ml_glaucoma.models import util


@gin.configurable(blacklist=['inputs', 'output_spec'])
def application(
        inputs, output_spec, training=None, application='ResNet50',
        weights='imagenet', pooling='avg', kwargs=None):
    if kwargs is None:
        kwargs = {}
    features = getattr(tf.keras.applications, application)(
            include_top=False, weights=weights, pooling=pooling,
            input_tensor=inputs, **kwargs).outputs
    probs = util.features_to_probs(features, output_spec)
    return probs
