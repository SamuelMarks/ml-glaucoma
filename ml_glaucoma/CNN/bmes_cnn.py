"""Trains a simple convnet on the BMES dataset.
"""

from __future__ import print_function

from os import path, makedirs

import certifi
import cv2
import h5py
import keras
import numpy as np
from keras import backend as K
from keras.applications import ResNet50
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from urllib3 import PoolManager

from ml_glaucoma import get_logger
from ml_glaucoma.utils.get_data import get_data

logger = get_logger(__file__.partition('.')[0])

K.set_image_data_format('channels_last')


def download(download_dir, force_new=False):
    http = PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())

    if not path.exists(download_dir):
        makedirs(download_dir)

    base = 'https://github.com/fchollet/deep-learning-models/releases/download'
    paths = '/v0.1/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',

    # TODO: Concurrency
    for fname in paths:
        basename = fname[fname.rfind('/') + 1:]
        to_file = path.join(download_dir, basename)

        if not force_new and path.isfile(to_file):
            logger.info('File exists: {to_file}'.format(to_file=to_file))
            continue

        logger.info('Downloading: "{basename}" to: "{download_dir}"'.format(
            basename=basename, download_dir=download_dir
        ))

        r = http.request('GET', '{base}{fname}'.format(base=base, fname=fname),
                         preload_content=False)
        with open(to_file, 'wb') as f:
            for chunk in r.stream(32):
                f.write(chunk)


def prepare_data(save_to, pixels, force_new=False):
    def _parse_function(filename):
        image = cv2.imread(filename)
        image_resized = cv2.resize(image, (pixels, pixels))
        print("Importing image ", prepare_data.i, end='\r')
        prepare_data.i += 1
        return image_resized

    def _get_filenames(neg_ids, pos_ids, id_to_imgs):
        # returns filenames list and labels list
        labels = []
        filenames = []
        for id in list(pos_ids) + list(neg_ids[:120]):
            for filename in id_to_imgs[id]:
                if id in pos_ids:
                    labels += [1]
                else:
                    labels += [0]
                filenames += [filename]
        return filenames, labels

    def _create_dataset(data_obj):
        pos_ids = data_obj.pickled_cache['oags1']
        neg_ids = data_obj.pickled_cache['no_oags1']
        id_to_imgs = data_obj.pickled_cache['id_to_imgs']

        img_names, data_labels = _get_filenames(neg_ids, pos_ids, id_to_imgs)

        print("Total images: ", len(img_names))

        prepare_data.i = 1
        dataset_tensor = np.stack(list(map(_parse_function, img_names)))
        print()

        return dataset_tensor, data_labels

    if not force_new and path.isfile(save_to):
        f = h5py.File(save_to, 'r')
        x_train_dset = f.get('x_train')
        y_train_dset = f.get('y_train')
        x_test_dset = f.get('x_test')
        y_test_dset = f.get('y_test')
        # X = numpy.array(Xdset)
        return (x_train_dset, y_train_dset), (x_test_dset, y_test_dset)

    data_obj = get_data()
    x, y = _create_dataset(data_obj)

    x, y = shuffle(x, y, random_state=0)
    x = x.astype('float32')
    x /= 255.

    # train_fraction = 0.9
    train_amount = int(x.shape[0] * 0.9)
    x_train, y_train = x[:train_amount], y[:train_amount]
    x_test, y_test = x[train_amount:], y[train_amount:]

    f = h5py.File(save_to, 'w')
    x_train = f.create_dataset("x_train", data=x_train, )  # compression='lzf')
    y_train = f.create_dataset("y_train", data=y_train, )  # compression='lzf')
    x_test = f.create_dataset("x_test", data=x_test, )  # compression='lzf')
    y_test = f.create_dataset("y_test", data=y_test, )  # compression='lzf')

    return (x_train, y_train), (x_test, y_test)


prepare_data.i = 1


# input image dimensions
# img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets


def run(download_dir, save_to, batch_size, num_classes, epochs,
        transfer_model, model_name, dropout, pixels):
    download(download_dir)

    (x_train, y_train), (x_test, y_test) = prepare_data(save_to, pixels)  # cifar10.load_data()
    print("Fraction negative training examples: ", (len(y_train) - np.sum(y_train)) / len(y_train))

    # indices = [i for i,label in enumerate(y_train) if label > 1]
    # y_train = np.delete(y_train,indices,axis=0)
    # x_train = np.delete(x_train,indices,axis=0)

    # indices = [i for i,label in enumerate(y_test) if label > 1]
    # y_test = np.delete(y_test,indices,axis=0)
    # x_test = np.delete(x_test,indices,axis=0)

    # if K.image_data_format() == 'channels_first':
    #    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    #    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    #    input_shape = (1, img_rows, img_cols)
    # else:
    #    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    #    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    #    input_shape = (img_rows, img_cols, 1)
    input_shape = x_train.shape[1:]

    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    resnet_weights_path = path.join(download_dir, 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

    model = Sequential()

    # TODO: Optic-disc segmentation at this point, or run optic-disc segmentation at this point

    if transfer_model == 'vgg16':
        vgg_model = keras.applications.vgg16.VGG16(weights='imagenet')
        vgg_model.summary()
        for layer in vgg_model.layers:
            model.add(layer)
    elif transfer_model == 'resnet50':
        model.add(ResNet50(include_top=False, pooling='avg'  # , weights=resnet_weights_path
                           ))
        model.add(Dense(num_classes, activation='softmax'))

        # Say not to train first layer (ResNet) model. It is already trained
        model.layers[0].trainable = False
    else:
        model.add(Conv2D(32,
                         kernel_size=(7, 7),  # as suggested
                         activation='relu',
                         padding='same',  # as suggested
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))  # as suggested
        if dropout > 3:
            model.add(Dropout(.5))  # as suggested
        model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))  # as suggested
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if dropout > 2:
            model.add(Dropout(.5))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))  # as suggested
        model.add(MaxPooling2D(pool_size=(2, 2)))  # as suggested
        if dropout > 1:
            model.add(Dropout(.5))  # as suggested
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        if dropout > 0:
            model.add(Dropout(.5))
        model.add(Dense(num_classes, activation='softmax'))

    # custom metrics for categorical
    def specificity(y_true, y_pred):
        return K.cast(K.all(
            (K.equal(K.argmax(y_true, axis=-1), 1), K.equal(K.argmax(y_pred, axis=-1), 1))
            , axis=1), K.floatx())

    def sensitivity(y_true, y_pred):
        return K.cast(K.all(
            (K.equal(K.argmax(y_true, axis=-1), 2), K.equal(K.argmax(y_pred, axis=-1), 2))
            , axis=1), K.floatx())

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              shuffle='batch',
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save model and weights
    save_dir = path.dirname(save_to)
    if not path.isdir(save_dir):
        makedirs(save_dir)
    model_path = path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at {}'.format(model_path))

    predictions = model.predict(x_test)
    y_test = np.argmax(y_test, axis=-1)
    predictions = np.argmax(predictions, axis=-1)
    confusion = confusion_matrix(y_test, predictions)
    print("Confusion matrix:")
    print(confusion)
    c = confusion
    print("sensitivity = ", c[0, 0] / (c[0, 1] + c[0, 0]))
    print("specificity = ", c[1, 1] / (c[1, 1] + c[1, 0]))