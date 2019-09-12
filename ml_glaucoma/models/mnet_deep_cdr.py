from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from inspect import currentframe

import gin
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D

from ml_glaucoma.models import util


@gin.configurable(blacklist=['inputs', 'output_spec'])
def unet(inputs, output_spec, training=None, kernel_regularizer=None,
         dropout_rate=0.3, hidden_activation='relu',
         final_activation='default'):
    if len(output_spec.shape) < 3:
        raise ValueError(
            'unet provides image outputs, but output_spec has shape %s'
            % (str(output_spec)))
    conv_kwargs = dict(
        kernel_regularizer=kernel_regularizer, padding='same',
        activation=hidden_activation)

    def concat(args, axis=-1):
        return tf.keras.layers.Lambda(
            tf.concat, arguments=dict(axis=axis))(args)

    conv1 = Conv2D(32, 3, 3, **conv_kwargs)(inputs)

    probs = util.features_to_probs(
        conv1, output_spec, layer_fn=Conv2D,
        kernel_regularizer=kernel_regularizer,
        activation=final_activation)

    model = tf.keras.models.Model(input=inputs, output=probs)
    model._name = currentframe().f_code.co_name

    return model
