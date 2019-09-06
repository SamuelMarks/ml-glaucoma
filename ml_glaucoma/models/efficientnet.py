from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import efficientnet.tfkeras as efficientnet_tfkeras_models
import gin
import tensorflow as tf


@gin.configurable(blacklist=['inputs', 'output_spec'])
def efficientnet(inputs, output_spec, num_classes=2, transfer_model=None,
                 image_size=224, num_channels=None):
    assert num_channels is not None

    assert transfer_model is not None and transfer_model in dir(
        efficientnet_tfkeras_models), '`transfer_model` not found'
    base_model = getattr(efficientnet_tfkeras_models, transfer_model)(
        input_shape=(image_size, image_size, num_channels),
        include_top=False,  # Set to false to train
        weights='imagenet'
    )
    base_model.trainable = True

    return tf.keras.Sequential([
        inputs,
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])
