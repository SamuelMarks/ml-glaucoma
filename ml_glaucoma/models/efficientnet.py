from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB0


@gin.configurable(blacklist=['inputs', 'output_spec'])
def efficientnet_b0(inputs, output_spec, num_classes=2,
                    image_size=224, num_channels=None):
    assert num_channels is not None

    base_model = EfficientNetB0(input_shape=(image_size, image_size, num_channels),
                                include_top=False,  # Set to false to train
                                weights='imagenet')
    base_model.trainable = True

    return tf.keras.Sequential([
        inputs,
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])
