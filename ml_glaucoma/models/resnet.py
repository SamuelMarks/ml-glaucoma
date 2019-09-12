from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from inspect import currentframe

import gin
import tensorflow as tf


@gin.configurable(blacklist=['inputs', 'output_spec'])
def resnet50(inputs, output_spec, num_classes=2,
             image_size=224, num_channels=None):
    assert num_channels is not None
    x = inputs

    base_model = tf.keras.applications.ResNet50(input_shape=(image_size, image_size, num_channels),
                                                include_top=False,  # Set to false to train
                                                weights='imagenet')
    base_model.trainable = True

    model = tf.keras.Sequential([
        x,
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])
    '''
    model.outputs = util.features_to_probs(
        x, output_spec, kernel_regularizer=kernel_regularizer,
        activation=final_activation)
    '''
    model._name = currentframe().f_code.co_name
    return model
