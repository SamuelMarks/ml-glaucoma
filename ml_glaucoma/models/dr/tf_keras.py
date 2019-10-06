from inspect import currentframe

import gin
import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB5

import ml_glaucoma.models.utils.tf_keras
from ml_glaucoma.models import utils

# Based off https://www.kaggle.com/nemethpeti/keras-implementation-for-0-829-0-916#Model:-EfficientNet
@gin.configurable(blacklist=['inputs', 'output_spec'])
def dr0(inputs, output_spec,
        weights='imagenet', pooling='avg', final_activation='default',
        kwargs=None):
    if kwargs is None:
        kwargs = {}

    base = EfficientNetB5(weights=weights, include_top=False, pooling=pooling, **kwargs)
    base.trainable = True

    # dropout_dense_layer = 0.2 # for B0
    # dropout_dense_layer = 0.3 # for B3
    dropout_dense_layer = 0.4  # for B5

    features, = tf.keras.models.Sequential([
        inputs,
        base,
        tf.keras.layers.Dropout(dropout_dense_layer),
        tf.keras.layers.Dense(5, activation='softmax')
    ]).outputs
    probs = ml_glaucoma.models.utils.tf_keras.features_to_probs(
        features, output_spec, activation=final_activation)
    model = tf.keras.models.Model(inputs=inputs, outputs=probs)
    model._name = currentframe().f_code.co_name
    return model
