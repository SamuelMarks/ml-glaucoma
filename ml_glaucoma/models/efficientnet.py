from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import efficientnet.tfkeras as efficientnet_tfkeras_models
import gin
import tensorflow as tf

from ml_glaucoma.models import util
from ml_glaucoma.utils.helpers import get_upper_kv

efficientnet_models = get_upper_kv(efficientnet_tfkeras_models)


@gin.configurable(blacklist=['inputs', 'output_spec'])
def efficientnet(inputs, output_spec, num_classes=2, transfer_model=None,
                 image_size=224, num_channels=None):
    assert num_channels is not None

    assert transfer_model is not None and transfer_model in efficientnet_models, '`transfer_model` not found'
    base_model = efficientnet_models[transfer_model](
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


@gin.configurable(blacklist=['inputs', 'output_spec'])
def efficient_net(inputs, output_spec, application='EfficientNetB0',
                  weights='imagenet', pooling='avg', final_activation='default',
                  kwargs=None):
    assert application is not None and application in efficientnet_models, '`application` not found'

    if kwargs is None:
        kwargs = {}
    features, = efficientnet_models[application](
        include_top=False, weights=weights, pooling=pooling,
        input_tensor=inputs, **kwargs).outputs
    probs = util.features_to_probs(
        features, output_spec, activation=final_activation)
    return tf.keras.models.Model(inputs=inputs, outputs=probs)
