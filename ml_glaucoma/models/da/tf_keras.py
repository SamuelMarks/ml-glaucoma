from inspect import currentframe

import gin
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import ZeroPadding2D

import ml_glaucoma.models.utils.tf_keras


@gin.configurable(blacklist=["inputs", "output_spec"])
def da0(
    inputs,
    output_spec,
    dropout_rate=0.5,
    conv_activation="relu",
    dense_activation="relu",
    kernel_regularizer=None,
    final_activation="softmax",
):
    model = Model(
        inputs=inputs,
        outputs=ml_glaucoma.models.utils.tf_keras.features_to_probs(
            Sequential(
                [
                    ZeroPadding2D((1, 1), input_shape=(112, 112, 1)),
                    Conv2D(filters=64, kernel_size=(3, 3), activation=conv_activation),
                    ZeroPadding2D((1, 1)),
                    Conv2D(filters=64, kernel_size=(3, 3), activation=conv_activation),
                    MaxPooling2D((2, 2), strides=(2, 2)),
                    ZeroPadding2D((1, 1)),
                    Conv2D(128, kernel_size=(3, 3), activation=conv_activation),
                    ZeroPadding2D((1, 1)),
                    Conv2D(128, kernel_size=(3, 3), activation=conv_activation),
                    MaxPooling2D((2, 2), strides=(2, 2)),
                    ZeroPadding2D((1, 1)),
                    Conv2D(256, kernel_size=(3, 3), activation=conv_activation),
                    ZeroPadding2D((1, 1)),
                    Conv2D(256, kernel_size=(3, 3), activation=conv_activation),
                    ZeroPadding2D((1, 1)),
                    Conv2D(256, kernel_size=(3, 3), activation=conv_activation),
                    MaxPooling2D((2, 2), strides=(2, 2)),
                    ZeroPadding2D((1, 1)),
                    Conv2D(512, kernel_size=(3, 3), activation=conv_activation),
                    ZeroPadding2D((1, 1)),
                    Conv2D(512, kernel_size=(3, 3), activation=conv_activation),
                    ZeroPadding2D((1, 1)),
                    Conv2D(512, kernel_size=(3, 3), activation=conv_activation),
                    MaxPooling2D((2, 2), strides=(2, 2)),
                    ZeroPadding2D((1, 1)),
                    Conv2D(512, kernel_size=(3, 3), activation=conv_activation),
                    ZeroPadding2D((1, 1)),
                    Conv2D(512, kernel_size=(3, 3), activation=conv_activation),
                    ZeroPadding2D((1, 1)),
                    Conv2D(512, kernel_size=(3, 3), activation=conv_activation),
                    MaxPooling2D((2, 2), strides=(2, 2)),
                    Flatten(),
                    Dense(4096, activation=dense_activation),
                    Dropout(dropout_rate),
                    Dense(4096, activation=dense_activation),
                    Dropout(dropout_rate),
                    Dense(2, activation=final_activation),
                ]
            ),
            output_spec,
            kernel_regularizer=kernel_regularizer,
            activation=final_activation,
        ),
    )
    model._name = currentframe().f_code.co_name
    return model


def da1(inputs, width, height, depth, classes):
    # initialize the model along with the input shape to be
    # "channels last" ordering
    model = Sequential()
    inputShape = (height, width, depth)
    # define the first (and only) CONV => RELU layer
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    # softmax classifier
    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    # return the constructed network architecture
    return model


del gin, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Sequential, ZeroPadding2D, Model

__all__ = ["da0", "da1"]
