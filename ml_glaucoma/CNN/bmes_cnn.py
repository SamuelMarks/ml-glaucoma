"""Trains a simple convnet on the BMES dataset.
"""

from __future__ import print_function

from itertools import groupby, takewhile, islice
from operator import itemgetter
from os import path, makedirs
from platform import python_version_tuple

import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from six import iteritems
from sklearn.metrics import precision_recall_fscore_support, fbeta_score, confusion_matrix

from ml_glaucoma.utils import pp

if python_version_tuple()[0] == '3':
    from functools import reduce

    imap = map
    ifilter = filter

import numpy as np

from keras.callbacks import Callback

from ml_glaucoma import get_logger, __version__
from ml_glaucoma.utils.get_data import get_data

K.set_image_data_format('channels_last')

logger = get_logger(__file__.partition('.')[0])


def parser(infile, top, threshold, by_diff):
    epoch2stat = {
        key: val
        for key, val in iteritems(
        {
            k: tuple(imap(itemgetter(1), v))
            for k, v in groupby(
            imap(lambda l: (l[0], l[1]),
                 ifilter(None, imap(
                     lambda l: (lambda fst: (
                         lambda three: (int(three), l.rstrip()[l.rfind(':') + 2:])
                         if three is not None and three.isdigit() and int(three[0]) < 4 else None)(
                         l[fst - 3:fst] if fst > -1 else None))(l.rfind(']')), infile)
                         ))
            , itemgetter(0))
        })
        if val and len(val) == 2
    }

    if threshold is not None:
        within_threshold = sorted((
            (k, reduce(lambda a, b: a >= threshold <= b, imap(
                lambda val: float(''.join(takewhile(lambda c: c.isdigit() or c == '.', val[::-1]))[::-1]),
                v))
             ) for k, v in iteritems(epoch2stat)), key=itemgetter(1))
        pp(tuple(islice((epoch2stat[k[0]] for k in within_threshold if k[1]), 0, top)))
    elif by_diff:
        lowest_diff = sorted((
            (k, reduce(lambda a, b: abs(a - b),
                       imap(lambda val: float(''.join(takewhile(lambda c: c.isdigit() or c == '.', val[::-1]))[::-1]),
                            v))
             ) for k, v in iteritems(epoch2stat)), key=itemgetter(1))

        pp(tuple(islice((epoch2stat[k[0]] for k in lowest_diff), 0, top)))
    else:
        pp(epoch2stat)


def download(download_dir, force_new=False):
    from urllib3 import PoolManager
    import certifi

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


def prepare_data(preprocess_to, pixels, force_new=False):
    import cv2
    import h5py
    from sklearn.utils import shuffle

    def _parse_function(filename):
        image = cv2.imread(filename)
        image_resized = cv2.resize(image, (pixels, pixels))
        print('Importing image ', prepare_data.i, end='\r')
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

        print('Total images: ', len(img_names))

        prepare_data.i = 1
        dataset_tensor = np.stack(list(map(_parse_function, img_names)))
        print()

        return dataset_tensor, data_labels

    if not force_new and path.isfile(preprocess_to):
        f = h5py.File(preprocess_to, 'r')
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

    f = h5py.File(preprocess_to, 'w')
    x_train = f.create_dataset('x_train', data=x_train, )  # compression='lzf')
    y_train = f.create_dataset('y_train', data=y_train, )  # compression='lzf')
    x_test = f.create_dataset('x_test', data=x_test, )  # compression='lzf')
    y_test = f.create_dataset('y_test', data=y_test, )  # compression='lzf')

    return (x_train, y_train), (x_test, y_test)


prepare_data.i = 1


# input image dimensions
# img_rows, img_cols = 28, 28

def output_sensitivity_specificity(epoch, predictions, y_test):
    from sklearn.metrics import confusion_matrix

    y_test = np.argmax(y_test, axis=-1)
    predictions = np.argmax(predictions, axis=-1)
    c = confusion_matrix(y_test, predictions)
    print('Confusion matrix:\n', c)
    print('[{:03d}] sensitivity'.format(epoch), c[0, 0] / (c[0, 1] + c[0, 0]))
    print('[{:03d}] specificity'.format(epoch), c[1, 1] / (c[1, 1] + c[1, 0]))


# the data, shuffled and split between train and test sets
class SensitivitySpecificityCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch:
            x_test = self.validation_data[0]
            y_test = self.validation_data[1]
            predictions = self.model.predict(x_test)
            output_sensitivity_specificity(epoch, predictions, y_test)


# from: https://stackoverflow.com/a/48720556
def reweight(y_true, y_pred, tp_weight=0.2, tn_weight=0.2, fp_weight=1.2, fn_weight=1.2):
    # Get predictions
    y_pred_classes = K.greater_equal(y_pred, 0.5)
    y_pred_classes_float = K.cast(y_pred_classes, K.floatx())

    # Get misclassified examples
    wrongly_classified = K.not_equal(y_true, y_pred_classes_float)
    wrongly_classified_float = K.cast(wrongly_classified, K.floatx())

    # Get correctly classified examples
    correctly_classified = K.equal(y_true, y_pred_classes_float)
    correctly_classified_float = K.cast(wrongly_classified, K.floatx())

    # Get tp, fp, tn, fn
    tp = correctly_classified_float * y_true
    tn = correctly_classified_float * (1 - y_true)
    fp = wrongly_classified_float * y_true
    fn = wrongly_classified_float * (1 - y_true)

    # Get weights
    weight_tensor = tp_weight * tp + fp_weight * fp + tn_weight * tn + fn_weight * fn

    loss = K.binary_crossentropy(y_true, y_pred)
    weighted_loss = loss * weight_tensor
    return weighted_loss


def f_score_obj(y_true, y_pred):
    y_true = K.eval(y_true)
    y_pred = K.eval(y_pred)
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred)
    return K.variable(1. - f_score[1])


def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)


def binary_segmentation_recall(y_true, y_pred):
    # We assume y_true and y_pred have shape [?, W, H, 1]
    # This is a segmentation loss, for classification it doesn't really make sense.
    tp_ish = K.hard_sigmoid(100. * (y_pred - 0.5))
    approx_recall = K.sum(tp_ish, [1, 2, 3]) / K.sum(y_true, axis=[1, 2, 3])
    return approx_recall


class BinaryTruePositives(keras.layers.Layer):
    """Stateful Metric to count the total true positives over all batches.
        Assumes predictions and targets of shape `(samples, 1)`.
        # Arguments
            name: String, name for the metric.
    """

    def __init__(self, name='true_positives', **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.stateful = True
        self.true_positives = K.variable(value=0, dtype='int32')

    def reset_states(self):
        K.set_value(self.true_positives, 0)

    def __call__(self, y_true, y_pred):
        """Computes the number of true positives in a batch.
        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions
        # Returns
            The total number of true positives seen this epoch at the
                completion of the batch.
        """
        y_true = K.cast(y_true, 'int32')
        y_pred = K.cast(K.round(y_pred), 'int32')
        correct_preds = K.cast(K.equal(y_pred, y_true), 'int32')
        true_pos = K.cast(K.sum(correct_preds * y_true), 'int32')
        current_true_pos = self.true_positives * 1
        self.add_update(K.update_add(self.true_positives,
                                     true_pos),
                        inputs=[y_true, y_pred])

        return current_true_pos + true_pos


# custom metrics for categorical
def specificity(y_true, y_pred):
    return K.cast(K.all((
        K.equal(K.argmax(y_true, axis=-1), 1),
        K.equal(K.argmax(y_pred, axis=-1), 1)
    ), axis=1), K.floatx())


def sensitivity(y_true, y_pred):
    return K.cast(K.all((
        K.equal(K.argmax(y_true, axis=-1), 2),
        K.equal(K.argmax(y_pred, axis=-1), 2)
    ), axis=1), K.floatx())


def specificity_at_sensitivity(sensitivity, **kwargs):
    def metric(labels, predictions):
        # any tensorflow metric
        value, update_op = tf.metrics.specificity_at_sensitivity(labels, predictions, sensitivity, **kwargs)

        # find all variables created for this metric
        metric_vars = (i for i in tf.local_variables() if 'specificity_at_sensitivity' in i.name.split('/')[2])

        # Add metric variables to GLOBAL_VARIABLES collection.
        # They will be initialized for new session.
        for v in metric_vars:
            tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

        # force to update metric values
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
            return value

    return metric


def run(download_dir, preprocess_to, batch_size, num_classes, epochs,
        transfer_model, model_name, dropout, pixels, tensorboard_log_dir,
        optimizer, loss):
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
          'Uses optimiser: {optimizer} with loss: {loss}'.format(version=__version__,
                                                                 transfer_model=transfer_model,
                                                                 dropout=dropout,
                                                                 optimizer=optimizer,
                                                                 loss=loss))

    (x_train, y_train), (x_test, y_test) = prepare_data(preprocess_to, pixels)  # cifar10.load_data()
    print('Fraction negative training examples:', np.divide(np.subtract(len(y_train), np.sum(y_train)), len(y_train)))

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

    metric_fn = BinaryTruePositives()
    config = keras.metrics.serialize(metric_fn)
    metric_fn = keras.metrics.deserialize(
        config, custom_objects={'BinaryTruePositives': BinaryTruePositives})

    model.compile(loss=getattr(keras.losses, loss),
                  optimizer=getattr(keras.optimizers, optimizer)() if optimizer in dir(keras.optimizers) else optimizer,
                  metrics=['accuracy', metric_fn])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              shuffle='batch',
              validation_data=(x_test, y_test),
              callbacks=callbacks)
    score = model.evaluate(x_test, y_test, verbose=0)

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
