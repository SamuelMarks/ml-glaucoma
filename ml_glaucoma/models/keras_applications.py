from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from inspect import currentframe

import gin
import tensorflow as tf


@gin.configurable(blacklist=['inputs', 'output_spec'])
def keras_applications(inputs, output_spec, num_classes=2,
                       transfer_model=None, weights=None,
                       image_size=224, num_channels=None):
    assert num_channels is not None
    assert transfer_model is not None and transfer_model in dir(tf.keras.applications), '`transfer_model` not found'
    base_model = getattr(tf.keras.applications, transfer_model)(input_shape=(image_size, image_size, num_channels),
                                                                include_top=False,  # Set to false to train
                                                                weights=weights)
    base_model.trainable = True

    model = tf.keras.Sequential([
        inputs,
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])
    model._name = '_'.join((currentframe().f_code.co_name, transfer_model))
    return model
