"""Trains a simple convnet on the BMES dataset.
"""

from __future__ import print_function

import logging
from functools import partial
from itertools import chain
from os import path, makedirs, listdir
from platform import python_version_tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K, Input, Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (Dense, Dropout, Flatten, Activation,
                                     MaxPooling2D, Conv2D, UpSampling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from ml_glaucoma import get_logger, __version__
from ml_glaucoma.CNN.directories2tfrecords import convert_to_tfrecord
from ml_glaucoma.CNN.helpers import output_sensitivity_specificity
from ml_glaucoma.CNN.loss import weighted_categorical_crossentropy
from ml_glaucoma.CNN.metrics import BinaryTruePositives, Recall, Precision, binary_segmentation_recall
from ml_glaucoma.CNN.test0 import test0
from ml_glaucoma.utils.get_data import get_data

# from tensorflow.python.platform import tf_logging

if python_version_tuple()[0] == '3':
    xrange = range
    izip = zip
    imap = map
    basestring = str
else:
    from itertools import izip, imap

K.set_image_data_format('channels_last')

logger = get_logger(__file__.partition('.')[0])
logger.setLevel(logging.CRITICAL)
# tf_logging._get_logger().setLevel(logging.CRITICAL)
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
        optimizer, loss, architecture, metrics, split_dir, class_mode, lr, max_imgs):
    """

    :param download_dir: Directory to store precompiled CNN nets
    :type  download_dir: ``str``

    :param bmes123_pardir: Parent folder of BMES123 folder
    :type  bmes123_pardir: ``str``

    :param preprocess_to: Save h5 file of dataset, following preprocessing
    :type  preprocess_to: ``str``

    :param batch_size: Batch size
    :type  batch_size: ``int``

    :param num_classes: Number of classes
    :type  num_classes: ``int``

    :param epochs: Number of epochs
    :type  epochs: ``int``

    :param transfer_model: Transfer model. Currently any one of: `keras.application`, e.g.: "vgg16"; "resnet50"
    :type  transfer_model: ``str``

    :param model_name: Filename for h5 trained model file
    :type  model_name: ``str``

    :param dropout: Dropout (0,1,2,3 or 4)
    :type  dropout: ``int``

    :param pixels: Pixels. E.g.: 400 for 400px * 400px
    :type  pixels: ``int``

    :param tensorboard_log_dir: Enables Tensorboard integration and sets its log dir
    :type  tensorboard_log_dir: ``str``

    :param optimizer:
    :type  optimizer: ``str``

    :param loss:
    :type  loss: ``str``

    :param architecture: Current options: unet; for U-Net architecture
    :type  architecture: ``str``

    :param metrics: precision_recall or btp
    :type  metrics: ``str``

    :param split_dir: Place to create symbolic links for train, test, validation split
    :type  split_dir: ``str``

    :param class_mode: Determines the type of label arrays that are returned
    :type  class_mode: ``str``

    :param lr: Learning rate of optimiser
    :type  lr: ``int``

    :param max_imgs:
    :type  max_imgs: ``int``

    :return:
    """
    print('\n============================\nml_glaucoma {version} with transfer of {transfer_model} (dropout: {dropout}.'
          ' Uses optimiser: {optimizer} with loss: {loss})'.format(version=__version__,
                                                                   transfer_model=transfer_model,
                                                                   dropout=dropout,
                                                                   optimizer=optimizer,
                                                                   loss=loss))

    if class_mode == 'binary':
        num_classes = 1
        channels = 3
        activation = 'softmax'
    else:
        num_classes = 2
        channels = 3
        activation = 'softmax'

    # download(download_dir)

    save_dir = path.dirname(preprocess_to)
    if not path.isdir(save_dir):
        makedirs(save_dir)

    assert model_name != path.basename(preprocess_to), '{model_name} is same as {preprocess_to}'.format(
        model_name=model_name, preprocess_to=preprocess_to
    )

    test_dir, train_dir, validation_dir = get_data(base_dir=bmes123_pardir, split_dir=split_dir, max_imgs=max_imgs)
    partitions = 'test', 'train', 'validation'

    class_names = listdir(train_dir)  # Get names of classes
    class_name2id = {label: index for index, label in enumerate(class_names)}  # Map class names to integer labels

    tfrecords_dir = path.join(path.dirname(split_dir), 'tfrecords')
    if not path.isdir(tfrecords_dir):
        makedirs(tfrecords_dir)

    get_files = lambda directory: tuple(chain(
        imap(lambda cls: imap(lambda p: path.join(directory, cls, p), listdir(path.join(directory, cls))), class_names)
    ))

    tfrecord_fn = lambda curr_partition, partition_seq: convert_to_tfrecord(
        dataset_name=partition_seq,
        data_directory=path.join(tfrecords_dir, curr_partition),
        files=get_files(partition_seq),
        class_map=class_name2id,
        directories_as_labels=True, pixels=pixels
    )

    for partition in partitions:
        tfrecord_fn(partition, locals()['{}_dir'.format(partition)])

    exit(5)

    idg = ImageDataGenerator()  # (horizontal_flip=True, vertical_flip=True)

    flow = partial(idg.flow_from_directory, color_mode={1: 'grayscale', 3: 'rgb'}[channels],
                   target_size=(pixels, pixels), shuffle=True, class_mode=class_mode, follow_links=True)

    test0(train_dir=train_dir, validation_dir=validation_dir, test_dir=test_dir, pixels=pixels)

    # train_generator = idg.flow_from_directory(train_dir, color_mode={1: 'grayscale', 3: 'rgb'} [channels],
    #                                          target_size=(pixels, pixels), shuffle = True, class_mode=class_mode,
    #                                          follow_links=True)

    # flow(directory=train_dir)
    train_seq = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
        samplewise_std_normalization=True
    )
    valid_seq = flow(directory=validation_dir)  # type: tf.keras.preprocessing.image.DirectoryIterator
    test_seq = flow(directory=test_dir)  # type: tf.keras.preprocessing.image.DirectoryIterator

    mk_dataset = lambda seq: tf.data.Dataset.from_generator(lambda: seq, (tf.float32, tf.float32))
    # \type: (ImageDataGenerator) => tf.data.Dataset

    train_dataset = mk_dataset(train_seq)  # type: tf.data.Dataset
    valid_dataset = mk_dataset(valid_seq)  # type: tf.data.Dataset
    test_dataset = mk_dataset(test_seq)  # type: tf.data.Dataset

    print('train_dataset:', train_dataset, ';')

    # dataset = tf.data.Dataset.from_generator(train_seq)
    # print('dataset:', dataset)

    callbacks = [
        # SensitivitySpecificityCallback(validation_data=valid_seq, class_mode=class_mode)]
    ]
    # tf.metrics.auc(num_thresholds=3)

    if tensorboard_log_dir:
        if not path.isdir(tensorboard_log_dir):
            makedirs(tensorboard_log_dir)
        callbacks.append(
            TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=0, write_graph=True, write_images=True)
        )

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
        input_shape = channels, pixels, pixels
    else:
        #    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        #    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = pixels, pixels, channels
    # input_shape = x_train.shape[1:]

    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    # print('x_train shape:', x_train.shape)
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    # y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    # y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # resnet_weights_path = path.join(download_dir, 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

    model = Sequential()

    # TODO: Optic-disc segmentation at this point, or run optic-disc segmentation at this point

    if transfer_model is not None:
        transfer_model_lower = transfer_model.lower()
        if transfer_model_lower.startswith('vgg') or transfer_model_lower.startswith('resnet'):
            model.add(getattr(keras.applications,
                              transfer_model.upper() if transfer_model_lower.startswith('vgg') else transfer_model)(
                include_top=False, weights='imagenet', pooling='avg'
            ))

        else:
            model.add(getattr(keras.applications, transfer_model)(
                include_top=False, pooling='avg'
            ))

        model.add(Dense(num_classes, activation=activation))

        model.layers[0].trainable = False
    else:
        if architecture == 'unet':
            model = get_unet_light_for_fold0(pixels, pixels)
        elif architecture == 'dc0':
            model.add(Conv2D(32, (3, 3), input_shape=input_shape))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(32, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            # this converts our 3D feature maps to 1D feature vectors
            model.add(Flatten(data_format=K.image_data_format()))

            model.add(Dense(64))  # we now have numbers not 'images'
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
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
            model.add(Conv2D(32, 3, activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            if dropout > 1:
                model.add(Dropout(.5))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            if dropout > 0:
                model.add(Dropout(.5))
        model.add(Dense(num_classes))
        model.add(Activation(activation))

    if metrics == 'precision_recall':
        metric_fn = BinaryTruePositives()
        config = tf.keras.metrics.serialize(metric_fn)
        metric_fn = tf.keras.metrics.deserialize(
            config, custom_objects={'BinaryTruePositives': BinaryTruePositives})
        metrics = ['acc', Recall(), Precision(), metric_fn]
    else:  # btp
        metric_fn = BinaryTruePositives()
        config = tf.keras.metrics.serialize(metric_fn)
        metric_fn = tf.keras.metrics.deserialize(
            config, custom_objects={'BinaryTruePositives': BinaryTruePositives})
        metrics = ['accuracy', metric_fn]

    if loss == 'weighted_categorical_crossentropy':
        loss = weighted_categorical_crossentropy(np.ones((num_classes, num_classes)))
    elif loss == 'binary_segmentation_recall':
        loss = binary_segmentation_recall

    model.compile(loss=loss if callable(loss) else getattr(tf.keras.losses, loss),
                  optimizer=getattr(tf.keras.optimizers, optimizer)(**({} if lr is None else {'lr': lr})),
                  metrics=metrics)
    print(model.summary())

    # x_val, y_val = izip(*(np.vstack(valid_seq[i]) for i in xrange(len(valid_seq))))
    x, y = izip(*(valid_seq[i] for i in xrange(len(valid_seq))))

    train_x, train_y = izip(*(train_seq[i] for i in xrange(len(train_seq))))

    '''
    np.save('/tmp/x_{}'.format(class_mode), x)
    np.save('/tmp/y_{}'.format(class_mode), y)
    '''

    x_val = np.vstack(x)
    y_val = np.vstack(imap(to_categorical, y))[:, 0] if class_mode == 'binary' else y

    print(type(y_val), y_val)

    metrics = [tf.metrics.auc(labels=x_val, predictions=y_val)]

    # x_train_val = np.stack(train_x)
    # y_train_val = np.vstack(imap(to_categorical, y))[:, 0] if class_mode == 'binary' else y

    print('x_val:', x_val, ';')
    print('y_cal:', y_val, ';')

    if train_seq.batch_size > train_seq.n:
        raise TypeError('train_seq.batch_size > train_seq.n')

    STEP_SIZE_TRAIN = train_seq.n // train_seq.batch_size

    if valid_seq.batch_size > valid_seq.n:
        raise TypeError('train_seq.batch_size > train_seq.n')

    STEP_SIZE_VALID = valid_seq.n // valid_seq.batch_size

    model.fit_generator(train_seq, validation_data=valid_seq,  # valid_generator,
                        epochs=epochs,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_steps=STEP_SIZE_VALID,
                        callbacks=callbacks, verbose=1)
    score = model.evaluate_generator(test_seq, verbose=0)

    '''model.fit_generator(train_dataset, validation_data=(x_val, y_val), # validation_data=valid_dataset,
                        epochs=epochs, steps_per_epoch=batch_size,
                        callbacks=callbacks, verbose=1)
    score = model.evaluate_generator(test_dataset, verbose=0)'''

    '''
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              shuffle='batch',
              validation_data=(x_test, y_test),
              callbacks=callbacks)
    score = model.evaluate(x_test, y_test, verbose=0)
    '''

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # print('specificity_at_sensitivity', specificity_at_sensitivity(x_test=x_test, y_test=y_test))

    # Save model and weights

    model_path = path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at "{model_path}"'.format(model_path=model_path))

    x_test, y_test = izip(*(np.vstack(test_seq[i])
                            for i in xrange(len(test_seq))))

    predictions = model.predict(x_test)
    y_test = np.argmax(y_test, axis=-1)

    predictions = np.argmax(predictions, axis=-1)

    predictions = np.argmax(predictions, axis=-1)
    confusion = confusion_matrix(y_test, predictions)
    print('Confusion matrix:')
    print(confusion)
    c = confusion
    print('sensitivity =', c[0, 0] / (c[0, 1] + c[0, 0]))
    print('specificity =', c[1, 1] / (c[1, 1] + c[1, 0]))

    confusion = tf.confusion_matrix(y_test, predictions)
    print('Confusion matrix:')
    print(confusion)
    c = confusion
    print('sensitivity =', c[0, 0] / (c[0, 1] + c[0, 0]))

    print('specificity =', c[1, 1] / (c[1, 1] + c[1, 0]))

    output_sensitivity_specificity(epochs, predictions, y_test, class_mode=class_mode)
