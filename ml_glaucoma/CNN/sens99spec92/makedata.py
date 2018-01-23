'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import tensorflow as tf
import keras
from keras.datasets import cifar10
from ml_glaucoma.utils.get_data import get_data
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras.backend as K
import os
import numpy as np
import cv2 
import h5py
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

DATA_SAVE_LOCATION = '/mnt/datasets/balancedsplit100x100.hdf5'

def prepare_data():
    def _parse_function(filename):
        image = cv2.imread(filename)
        #if(image.shape != (2592,3888,3)):
        #    image = cv2.resize(image, (2592,3888))
        image = cv2.resize(image, (100,150))
        image = image[25:125,:]
        return image
    def _get_filenames(neg_ids,pos_ids,id_to_imgs):
        #returns filenames list and labels list
        train_labels = []
        train_filenames = []
        test_labels = []
        test_filenames = []
        train_fraction = 0.1
        pos_ids = list(pos_ids)
        neg_ids = list(neg_ids)
        train_ids = pos_ids[int(len(pos_ids)*train_fraction):] + neg_ids[:120]
        test_ids = pos_ids[:int(len(pos_ids)*train_fraction)] + neg_ids[500:600]
        for id in train_ids:
            for filename in id_to_imgs[id]:
                if id in pos_ids:
                    train_labels += [1]
                else:
                    train_labels += [0]
                train_filenames += [filename]
        for id in test_ids:
            for filename in id_to_imgs[id]:
                if id in pos_ids:
                    test_labels += [1]
                else:
                    test_labels += [0]
                test_filenames += [filename]
        
        return (train_filenames, train_labels), (test_filenames, test_labels)

    def _create_dataset(data_obj,test=False):
        pos_ids = data_obj.pickled_cache['oags1']
        neg_ids = data_obj.pickled_cache['no_oags1']
        id_to_imgs = data_obj.pickled_cache['id_to_imgs']

        (img_names,data_labels),(test_names,test_labels)=_get_filenames(neg_ids,pos_ids,id_to_imgs)
        img_names, data_labels = shuffle(img_names,data_labels,random_state=0)

        if(test):
            for img, label in zip(test_names, test_labels):
                yield _parse_function(img), label
        else:
            for img, label in zip(img_names, data_labels):
                yield _parse_function(img), label
    def _get_size_of_dataset(data_obj):
        pos_ids = data_obj.pickled_cache['oags1']
        neg_ids = data_obj.pickled_cache['no_oags1']
        id_to_imgs = data_obj.pickled_cache['id_to_imgs']

        (img_names, data_labels),(test_names,_) = _get_filenames(neg_ids,pos_ids,id_to_imgs)
        img_shape = _parse_function(img_names[0]).shape
        return (len(img_names),*img_shape,), (len(test_names),*img_shape)

    data_obj = get_data()
    train_shape,test_shape = _get_size_of_dataset(data_obj)
    
    f = h5py.File(DATA_SAVE_LOCATION,'w')
    x_train = f.create_dataset("x_train", train_shape,
            compression='lzf')
    y_train = f.create_dataset("y_train", (train_shape[0],),compression='lzf')
    x_test = f.create_dataset("x_test",test_shape
            ,compression='lzf')
    y_test = f.create_dataset("y_test", (test_shape[0],), compression='lzf')
    
    i = 0
    for image, label in _create_dataset(data_obj):
        x_train[i,:,:,:] = image
        y_train[i] = label
        print("\rLoaded training image %d "%(i), end='')
        i += 1
    print()
    i = 0
    for image, label in _create_dataset(data_obj,test=True):
        x_test[i,:,:,:] = image
        y_test[i] = label
        print("\rLoaded test image %d "%(i), end='')
        i += 1
    print()
    return (x_train, y_train),(x_test, y_test)


# The data, shuffled and split between train and test sets:
#(x_train, y_train), (x_test, y_test) = prepare_data()
(x_train, y_train),(x_test, y_test) = prepare_data()

print("Length of both train arrays")
print(len(y_train))
print(len(x_train))
print("max: ", np.max(y_train))
print("min: ", np.min(y_train))
print("xtrainshape: ", x_train.shape)

print("Mean: ", np.mean(x_train))
print("Std: ",np.std(x_train))
print("Max: ",np.max(x_train))
print("Min: ",np.min(x_train))



print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print("Num positive training examples: ", np.sum(y_train))
print("Fraction negative training examples: ", (len(y_train) - np.sum(y_train))/len(y_train))

print("dataset creation finished")
exit()

