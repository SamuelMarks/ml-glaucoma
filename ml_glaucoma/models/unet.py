from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gin
import tensorflow as tf
from ml_glaucoma.models import util

Conv2D = tf.keras.layers.Conv2D
Dropout = tf.keras.layers.Dropout
MaxPooling2D = tf.keras.layers.MaxPooling2D
UpSampling2D = tf.keras.layers.UpSampling2D


@gin.configurable(blacklist=['inputs', 'output_spec'])
def unet(
        inputs, output_spec, training=None, kernel_regularizer=None,
        dropout_rate=0.3, hidden_activation='relu'):
    conv_kwargs = dict(
        kernel_regularizer=kernel_regularizer, padding='same',
        activation=hidden_activation)

    def concat(args, axis=-1):
        return tf.keras.layers.Lambda(
            tf.concat, arguments=dict(axis=axis))(args)

    conv1 = Conv2D(32, 3, 3, **conv_kwargs)(inputs)
    conv1 = Dropout(dropout_rate)(conv1, training=training)
    conv1 = Conv2D(32, 3, 3, **conv_kwargs)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, 3, **conv_kwargs)(pool1)
    conv2 = Dropout(dropout_rate)(conv2, training=training)
    conv2 = Conv2D(64, 3, 3, **conv_kwargs)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, 3, 3, **conv_kwargs)(pool2)
    conv3 = Dropout(dropout_rate)(conv3, training=training)
    conv3 = Conv2D(64, 3, 3, **conv_kwargs)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(64, 3, 3, **conv_kwargs)(pool3)
    conv4 = Dropout(dropout_rate)(conv4, training=training)
    conv4 = Conv2D(64, 3, 3, **conv_kwargs)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(64, 3, 3, **conv_kwargs)(pool4)
    conv5 = Dropout(dropout_rate)(conv5, training=training)
    conv5 = Conv2D(64, 3, 3, **conv_kwargs)(conv5)

    up6 = concat([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(64, 3, 3, **conv_kwargs)(up6)
    conv6 = Dropout(dropout_rate)(conv6, training=training)
    conv6 = Conv2D(64, 3, 3, **conv_kwargs)(conv6)

    up7 = concat([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(64, 3, 3, **conv_kwargs)(up7)
    conv7 = Dropout(dropout_rate)(conv7, training=training)
    conv7 = Conv2D(64, 3, 3, **conv_kwargs)(conv7)

    up8 = concat([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, 3, 3, **conv_kwargs)(up8)
    conv8 = Dropout(dropout_rate)(conv8, training=training)
    conv8 = Conv2D(64, 3, 3, **conv_kwargs)(conv8)

    up9 = concat([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, 3, 3, **conv_kwargs)(up9)
    conv9 = Dropout(dropout_rate)(conv9, training=training)
    conv9 = Conv2D(32, 3, 3, **conv_kwargs)(conv9)

    probs = util.features_to_probs(
        conv9, output_spec, layer_fn=Conv2D,
        kernel_regularizer=kernel_regularizer, kernel_size=1)

    model = tf.keras.models.Model(input=inputs, output=probs)

    return model
