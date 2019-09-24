from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from inspect import currentframe

import gin
import tensorflow as tf
from tensorflow.python.keras.layers import (Conv2D, MaxPooling2D, Dense,
                                            Dropout, GlobalAveragePooling2D,
                                            GlobalMaxPooling2D, Flatten)

from ml_glaucoma.models import util

_poolers = {
    'avg': GlobalAveragePooling2D,
    'max': GlobalMaxPooling2D,
    'flatten': Flatten,
}


@gin.configurable(blacklist=['inputs', 'output_spec'])
def dc0(inputs, output_spec, training=None, filters=(32, 32, 64),
        dense_units=(64,), dropout_rate=0.5, conv_activation='relu',
        dense_activation='relu', kernel_regularizer=None,
        final_activation='default', pooling='flatten'):
    conv_kwargs = dict(
        kernel_regularizer=kernel_regularizer, activation=conv_activation)
    dense_kwargs = dict(
        kernel_regularizer=kernel_regularizer, activation=dense_activation)

    x = inputs
    for f in filters:
        x = Conv2D(f, (3, 3), **conv_kwargs)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = _poolers[pooling]()(x)

    for u in dense_units:
        x = Dense(u, **dense_kwargs)(x)
        x = Dropout(dropout_rate)(x, training=training)
    probs = util.features_to_probs(
        x, output_spec, kernel_regularizer=kernel_regularizer,
        activation=final_activation)
    model = tf.keras.models.Model(inputs=inputs, outputs=probs)
    model._name = currentframe().f_code.co_name
    return model


@gin.configurable(blacklist=['inputs', 'output_spec'])
def dc1(inputs, output_spec, training=None, dropout_rate=0.5,
        num_dropout_layers=4, kernel_regularizer=None,
        conv_activation='relu', final_activation='default', pooling='avg'):
    conv_kwargs = dict(
        kernel_regularizer=kernel_regularizer, activation=conv_activation,
        padding='same')
    x = inputs
    x = Conv2D(32, (7, 7), **conv_kwargs)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    if num_dropout_layers > 3:
        x = Dropout(dropout_rate)(x, training=training)
    x = Conv2D(64, (5, 5), **conv_kwargs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    if num_dropout_layers > 2:
        x = Dropout(dropout_rate)(x, training=training)
    x = Conv2D(32, 3, **conv_kwargs)(x)
    if num_dropout_layers > 1:
        x = Dropout(dropout_rate)(x, training=training)
    # x = Flatten(data_format=tf.keras.backend.image_data_format())(x)
    x = _poolers[pooling]()(x)
    x = Dense(
        128, kernel_regularizer=kernel_regularizer,
        activation=conv_activation)(x)
    if num_dropout_layers > 0:
        x = Dropout(dropout_rate)(x, training=training)(x)
    probs = util.features_to_probs(
        x, output_spec, kernel_regularizer=kernel_regularizer,
        activation=final_activation)
    model = tf.keras.models.Model(inputs=inputs, outputs=probs)
    model._name = currentframe().f_code.co_name
    return model


@gin.configurable(blacklist=['inputs', 'output_spec'])
def dc2(inputs, output_spec, training=None, filters=(32, 32, 64),
        dense_units=(32,), dropout_rate=0.5, conv_activation='relu',
        dense_activation='relu', kernel_regularizer=None,
        final_activation='default', pooling='flatten'):
    conv_kwargs = dict(
        strides=(2, 1), kernel_initializer='he_normal',
        kernel_regularizer=kernel_regularizer, activation=conv_activation)
    dense_kwargs = dict(
        kernel_regularizer=kernel_regularizer, activation=dense_activation)

    x = inputs
    for f in filters:
        x = Conv2D(f, (11, 7), **conv_kwargs)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = _poolers[pooling]()(x)

    for u in dense_units:
        x = Dense(u, **dense_kwargs)(x)
        x = Dropout(dropout_rate)(x, training=training)
    probs = util.features_to_probs(
        x, output_spec, kernel_regularizer=kernel_regularizer,
        activation=final_activation)
    model = tf.keras.models.Model(inputs=inputs, outputs=probs)
    model._name = currentframe().f_code.co_name
    return model
