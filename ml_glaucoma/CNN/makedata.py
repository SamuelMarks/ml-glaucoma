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

DATA_SAVE_LOCATION = '/mnt/datasets/100x100_dataset.hdf5'

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
        labels = []
        filenames = []
        for id in list(pos_ids)+list(neg_ids)[:200]:
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

        img_names, data_labels = _get_filenames(neg_ids,pos_ids,id_to_imgs)
        img_names, data_labels = shuffle(img_names,data_labels,random_state=0)

        for img, label in zip(img_names, data_labels):
            yield _parse_function(img), label
    def _get_size_of_dataset(data_obj):
        pos_ids = data_obj.pickled_cache['oags1']
        neg_ids = data_obj.pickled_cache['no_oags1']
        id_to_imgs = data_obj.pickled_cache['id_to_imgs']

        img_names, data_labels = _get_filenames(neg_ids,pos_ids,id_to_imgs)
        return (len(img_names),*_parse_function(img_names[0]).shape,)

    data_obj = get_data()
    shape = _get_size_of_dataset(data_obj)
    train_fraction = 0.9
    train_amount = int(shape[0]*train_fraction)
    test_amount = shape[0] - train_amount
    
    f = h5py.File(DATA_SAVE_LOCATION,'w')
    x_train = f.create_dataset("x_train", (train_amount,shape[1],shape[2],shape[3]),
            compression='lzf')
    y_train = f.create_dataset("y_train", (train_amount,),compression='lzf')
    x_test = f.create_dataset("x_test", (test_amount,shape[1],shape[2],shape[3]), compression='lzf')
    y_test = f.create_dataset("y_test", (train_amount,), compression='lzf')
    
    i = 0
    for image, label in _create_dataset(data_obj):
        if(i < train_amount):
            x_train[i,:,:,:] = image
            y_train[i] = label
        else:
            x_test[i-train_amount,:,:,:] = image
            y_test[i-train_amount] = label
        print("\rLoaded image %d of %d "%(i,shape[0]), end='')
        i += 1

    #x, y = _create_dataset(data_obj)

    #x_train, y_train = x[:train_amount],y[:train_amount]
    #x_test, y_test = x[train_amount:],y[train_amount:]

    #x_train = f.create_dataset("x_train", data=x_train, )#compression='lzf')
    #y_train = f.create_dataset("y_train", data=y_train,)# compression='lzf')
    #x_test = f.create_dataset("x_test", data=x_test, )#compression='lzf')
    #y_test = f.create_dataset("y_test", data=y_test, )#compression='lzf')

    return (x_train, y_train),(x_test, y_test)

batch_size = 256
num_classes = 2
epochs = 10
data_augmentation = False
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_glaucoma_trained_model.h5'
CIFAR = False
categorical = True 

    # The data, shuffled and split between train and test sets:
    #(x_train, y_train), (x_test, y_test) = prepare_data()
x_train, y_train = prepare_data()

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

