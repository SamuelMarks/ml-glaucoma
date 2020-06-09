from __future__ import print_function

from os import path
from platform import python_version_tuple

import cv2
import h5py
import numpy as np
from sklearn.utils import shuffle

from ml_glaucoma.constants import IMAGE_RESOLUTION
from ml_glaucoma.utils.bmes_data_prep import get_data

if python_version_tuple()[0] == '2':
    from itertools import imap as map


def prepare_data(preprocess_to, pixels, force_new=False):
    def _parse_function(filename):
        image = cv2.imread(filename)
        image_resized = cv2.resize(image, (pixels, pixels))
        print('Importing image\t{:04d}\t{}'.format(prepare_data.i, filename), end='\r')
        prepare_data.i += 1
        return image_resized

    def _get_filenames(neg_ids, pos_ids, id_to_imgs):
        # returns filenames list and labels list
        labels = []
        filenames = []
        for curr_id in list(pos_ids) + list(neg_ids[:120]):
            for filename in id_to_imgs[curr_id]:
                if curr_id in pos_ids:
                    labels += [1]
                else:
                    labels += [0]
                filenames += [filename]
        return filenames, labels

    def _create_dataset(data):
        pos_ids = data.pickled_cache['oags1']
        neg_ids = data.pickled_cache['no_oags1']
        id_to_imgs = data.pickled_cache['id_to_imgs']

        img_names, data_labels = _get_filenames(neg_ids, pos_ids, id_to_imgs)

        print('Total images:', len(img_names))

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
    x /= float(IMAGE_RESOLUTION[0])

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

__all__ = ['prepare_data']
