"""Trains a simple convnet on the BMES dataset.
"""

from __future__ import print_function

import logging
from os import path, makedirs

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K, Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPooling2D, Conv2D, UpSampling2D
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.python.platform import tf_logging

from ml_glaucoma import get_logger, __version__
from ml_glaucoma.CNN.helpers import output_sensitivity_specificity
from ml_glaucoma.CNN.metrics import BinaryTruePositives, SensitivitySpecificityCallback, Recall, Precision
from ml_glaucoma.utils.get_data import get_data

K.set_image_data_format('channels_last')

logger = get_logger(__file__.partition('.')[0])
logger.setLevel(logging.CRITICAL)
tf_logging._get_logger().setLevel(logging.CRITICAL)
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)


# input image dimensions
# img_rows, img_cols = 28, 28


def get_unet_light_for_fold0(img_rows, img_cols):
    inputs = Input((3, img_rows, img_cols))
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.3)(conv1)
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.3)(conv2)
    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.3)(conv3)
    conv3 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Dropout(0.3)(conv6)
    conv6 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Dropout(0.3)(conv7)
    conv7 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Dropout(0.3)(conv8)
    conv8 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Dropout(0.3)(conv9)
    conv9 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Conv2D(1, 1, 1, activation='sigmoid', border_mode='same')(conv9)
    # conv10 = Flatten()(conv10)

    model = Model(input=inputs, output=conv10)

    return model


def run(download_dir, bmes123_pardir, preprocess_to, batch_size, num_classes, epochs,
        transfer_model, model_name, dropout, pixels, tensorboard_log_dir,
        optimizer, loss, architecture, metrics, split_dir):
    callbacks = [SensitivitySpecificityCallback()]
    if tensorboard_log_dir:
        if not path.isdir(tensorboard_log_dir):
            makedirs(tensorboard_log_dir)
        callbacks.append(
            TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=0, write_graph=True, write_images=True)
        )

    # download(download_dir)

    save_dir = path.dirname(preprocess_to)
    if not path.isdir(save_dir):
        makedirs(save_dir)

    assert model_name != path.basename(preprocess_to), '{model_name} is same as {preprocess_to}'.format(
        model_name=model_name, preprocess_to=preprocess_to
    )

    print('\n============================\nml_glaucoma {version} with transfer of {transfer_model} (dropout: {dropout}.'
          ' Uses optimiser: {optimizer} with loss: {loss})'.format(version=__version__,
                                                                   transfer_model=transfer_model,
                                                                   dropout=dropout,
                                                                   optimizer=optimizer,
                                                                   loss=loss))

    'split-dir'
    test_dir, train_dir, validation_dir = get_data(base_dir=bmes123_pardir, split_dir=split_dir)

    idg = ImageDataGenerator(horizontal_flip=True)

    train_seq = idg.flow_from_directory(train_dir, target_size=(pixels, pixels), shuffle=True)
    valid_seq = idg.flow_from_directory(validation_dir, target_size=(pixels, pixels), shuffle=True)
    test_seq = idg.flow_from_directory(test_dir, target_size=(pixels, pixels), shuffle=True)

    '''
    (x_train, y_train), (x_test, y_test) = prepare_data(preprocess_to, pixels)  # cifar10.load_data()
    print('x_train:', x_train, ';')
    print('y_train:', y_train, ';')
    print('x_test:', x_train, ';')
    print('y_test:', y_train, ';')

    print('Fraction negative training examples:', np.divide(np.subtract(len(y_train), np.sum(y_train)), len(y_train)))
    '''

    # indices = [i for i,label in enumerate(y_train) if label > 1]
    # y_train = np.delete(y_train,indices,axis=0)
    # x_train = np.delete(x_train,indices,axis=0)

    # indices = [i for i,label in enumerate(y_test) if label > 1]
    # y_test = np.delete(y_test,indices,axis=0)
    # x_test = np.delete(x_test,indices,axis=0)

    if K.image_data_format() == 'channels_first':
        #    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        #    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (3, pixels, pixels)
    else:
        #    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        #    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (pixels, pixels, 3)
    # input_shape = x_train.shape[1:]

    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    # print('x_train shape:', x_train.shape)
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    # resnet_weights_path = path.join(download_dir, 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

    model = Sequential()

    # TODO: Optic-disc segmentation at this point, or run optic-disc segmentation at this point

    if transfer_model is not None:
        transfer_model = transfer_model.lower()
        if transfer_model.startswith('vgg'):
            model.add(getattr(keras.applications, transfer_model.upper())(
                include_top=False, weights='imagenet', pooling='avg'
            ))

        else:
            model.add(getattr(keras.applications, transfer_model.upper())(
                include_top=False, pooling='avg'
            ))

        model.add(Dense(num_classes, activation='softmax'))

        model.layers[0].trainable = False
    else:
        if architecture == 'unet':
            model = get_unet_light_for_fold0(pixels, pixels)
        else:
            model.add(Conv2D(32,
                             kernel_size=(7, 7),
                             activation='relu',
                             padding='same',
                             input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            if dropout > 3:
                model.add(Dropout(.5))
            model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            if dropout > 2:
                model.add(Dropout(.5))
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            if dropout > 1:
                model.add(Dropout(.5))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            if dropout > 0:
                model.add(Dropout(.5))
        model.add(Dense(num_classes, activation='softmax'))

    if metrics == 'precision_recall':
        metric_fn = BinaryTruePositives()
        config = keras.metrics.serialize(metric_fn)
        metric_fn = keras.metrics.deserialize(
            config, custom_objects={'BinaryTruePositives': BinaryTruePositives})
        metrics = ['acc',
                   Recall(),
                   Precision(),
                   metric_fn]
    else:  # btp
        metric_fn = BinaryTruePositives()
        config = keras.metrics.serialize(metric_fn)
        metric_fn = keras.metrics.deserialize(
            config, custom_objects={'BinaryTruePositives': BinaryTruePositives})
        metrics = ['accuracy', metric_fn]

    model.compile(loss=getattr(keras.losses, loss),
                  optimizer=getattr(keras.optimizers, optimizer)() if optimizer in dir(keras.optimizers) else optimizer,
                  metrics=metrics)
    model.fit_generator(train_seq, validation_data=valid_seq, epochs=epochs, callbacks=callbacks, verbose=1,
                        batch_size=batch_size)
    score = model.evaluate_generator(test_seq, verbose=0)

    '''model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              shuffle='batch',
              validation_data=(x_test, y_test),
              callbacks=callbacks)
    score = model.evaluate(x_test, y_test, verbose=0)'''

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # print('specificity_at_sensitivity', specificity_at_sensitivity(x_test=x_test, y_test=y_test))

    # Save model and weights

    model_path = path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at "{model_path}"'.format(model_path=model_path))

    predictions = model.predict(x_test)
    y_test = np.argmax(y_test, axis=-1)

    predictions = np.argmax(predictions, axis=-1)

    predictions = np.argmax(predictions, axis=-1)
    confusion = confusion_matrix(y_test, predictions)
    print("Confusion matrix:")
    print(confusion)
    c = confusion
    print("sensitivity = ", c[0, 0] / (c[0, 1] + c[0, 0]))
    print("specificity = ", c[1, 1] / (c[1, 1] + c[1, 0]))

    confusion = tf.confusion_matrix(y_test, predictions)
    print("Confusion matrix:")
    print(confusion)
    c = confusion
    print("sensitivity = ", c[0, 0] / (c[0, 1] + c[0, 0]))

    print("specificity = ", c[1, 1] / (c[1, 1] + c[1, 0]))

    output_sensitivity_specificity(epochs, predictions, y_test)
