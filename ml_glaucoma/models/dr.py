from inspect import currentframe
from os import environ

import gin

from ml_glaucoma.models import util

if environ['TF']:
    import tensorflow as tf
    from efficientnet.tfkeras import EfficientNetB5


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
        probs = util.features_to_probs(
            features, output_spec, activation=final_activation)
        model = tf.keras.models.Model(inputs=inputs, outputs=probs)
        model._name = currentframe().f_code.co_name
        return model
elif environ['TORCH']:
    @gin.configurable(blacklist=['inputs', 'output_spec'])
    def dr0(inputs, output_spec,
            weights='imagenet', pooling='avg', final_activation='default',
            kwargs=None):
        name = currentframe().f_code.co_name
        raise NotImplementedError(name)
else:
    @gin.configurable(blacklist=['inputs', 'output_spec'])
    def dr0(inputs, output_spec,
            weights='imagenet', pooling='avg', final_activation='default',
            kwargs=None):
        name = currentframe().f_code.co_name
        raise NotImplementedError(name)
