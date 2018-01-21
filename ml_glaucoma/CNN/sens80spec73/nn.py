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
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import keras.backend as K
import os
import numpy as np
import cv2 
import h5py
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

DATA_SAVE_LOCATION = '/mnt/datasets/100x100_dataset.hdf5'

def prepare_data():
    if(os.path.isfile(DATA_SAVE_LOCATION)):
        f = h5py.File(DATA_SAVE_LOCATION,'r')
        x_train_dset = f.get('x_train')
        y_train_dset = f.get('y_train')
        x_test_dset = f.get('x_test')
        y_test_dset = f.get('y_test')
        return (x_train_dset,y_train_dset),(x_test_dset,y_test_dset)
    else:
        print("Data file isn't there")
        exit()


batch_size = 256
num_classes = 2
epochs = 100
data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_glaucoma_trained_model.h5'
CIFAR = False
categorical =True 

if not CIFAR:
    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = prepare_data()
else:
    print("Using CIFAR10 dataset")
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    indices = [i for i,label in enumerate(y_train) if label > 1]
    y_train = np.delete(y_train,indices,axis=0)
    x_train = np.delete(x_train,indices,axis=0)

    indices = [i for i,label in enumerate(y_test) if label > 1]
    y_test = np.delete(y_test,indices,axis=0)
    x_test = np.delete(x_test,indices,axis=0)


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
print("Num positive test examples: ", np.sum(y_test))
print("Fraction negative test examples: ", (len(y_test) - np.sum(y_test))/len(y_test))

# Convert class vectors to binary class matrices.
if categorical:
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
#model.add(InputLayer(input_tensor=x_train, input_shape=(None,200,200,3)))
model.add(BatchNormalization(
                input_shape=x_train.shape[1:],
                ))
#model.add(Conv2D(32, (3, 3), padding='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(32, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))

#model.add(Conv2D(64, (3, 3), padding='same'))
#model.add(Activation('relu'))
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
if categorical:
    model.add(Dense(2))
    model.add(Activation('softmax'))
else:
    model.add(Dense(1))


#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#opt = keras.optimizers.RMSprop()
opt = keras.optimizers.Adam()
#opt = keras.optimizers.SGD(lr=0.0001, decay = 1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy',
          optimizer=opt,
          metrics=['categorical_accuracy'])

print("Shape is: ",x_train.shape)
print("Label shape: ", y_train.shape) 

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.09,
              shuffle='batch',
              #class_weight={0:1.,1:1000.},
              )
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
#                        validation_split=0.09,
                        workers=4,
                        #class_weight={0:1,1:24},
                        )

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print("\nTest results\n")
for metric, score in zip(model.metrics_names, scores):
    print(metric, ':', score)

with open('log.txt', 'a') as f:
    print("\nTest results\n",file=f)
    for metric, score in zip(model.metrics_names, scores):
        print(metric, ':', score, file=f)
    
results = model.predict(x_test)

tn,fp,fn,tp = confusion_matrix(np.argmax(y_test,axis=1), np.argmax(results,axis=1)).ravel()
print("sensitivity:", tp/(tp+fp+0.00001))
print("specificity:", tn/(tn+fn+0.00001))
print("accuracy:", (tn+tp)/(tn+tp+fn+fp))

