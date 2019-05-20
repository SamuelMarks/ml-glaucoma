from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def features_to_probs(
        x, output_spec, layer_fn=tf.keras.layers.Dense,
        **layer_kwargs):
    if output_spec.shape == ():
        # single target
        probs = layer_fn(1, activation='sigmoid', **layer_kwargs)(x)
        # channels_axis = (
        #     -1 if tf.keras.backend.image_data_format() == 'channels_last'
        #     else 1)
        # probs = tf.keras.layers.Lambda(
        #     tf.squeeze, arguments=dict(axis=channels_axis))(x)
    else:
        num_classes = output_spec.shape[-1]
        activation = 'sigmoid' if num_classes == 1 else 'softmax'
        probs = layer_fn(
            num_classes, activation=activation, **layer_kwargs)(x)
    return probs
