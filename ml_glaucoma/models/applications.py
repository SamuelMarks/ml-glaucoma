from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin
from ml_glaucoma.models import util


@gin.configurable(blacklist=['inputs', 'output_spec'])
def applications_model(
        inputs, output_spec, application='ResNet50',
        weights='imagenet', pooling='avg', final_activation='default',
        kwargs=None):
    if kwargs is None:
        kwargs = {}
    features, = getattr(tf.keras.applications, application)(
            include_top=False, weights=weights, pooling=pooling,
            input_tensor=inputs, **kwargs).outputs
    probs = util.features_to_probs(
        features, output_spec, activation=final_activation)
    return tf.keras.models.Model(inputs=inputs, outputs=probs)
