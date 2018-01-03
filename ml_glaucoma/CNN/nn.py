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

def prepare_data(data_obj):
    def _parse_function(filename):
        image = cv2.imread(filename)
        image_resized = cv2.resize(image, (400,400))
        global i
        print("Imported image: ", i, end='\r')
        i += 1
        return image_resized
    def _get_filenames(dataset,pos_ids,id_to_imgs):
        #returns filenames list and labels list
        labels = []
        filenames = []
        for id in dataset:
            for filename in id_to_imgs[id]:
                if id in pos_ids:
                    labels += [1]
                else:
                    labels += [0]
                filenames += [filename]
        return filenames, labels

    def _create_dataset(data_obj, dataset,size=None):
        pos_ids = data_obj.pickled_cache['oags1']
        id_to_imgs = data_obj.pickled_cache['id_to_imgs']

        img_names, data_labels = _get_filenames(dataset,pos_ids,id_to_imgs)
        if size:
            img_names, data_labels = img_names[:size], data_labels[:size]

        global i
        i = 0
        dataset_tensor = np.stack(list(map(_parse_function,img_names)))
        print()

        return dataset_tensor, data_labels

    x_train, y_train = train = _create_dataset(data_obj, data_obj.datasets.train)
    x_val, y_val = validation = _create_dataset(data_obj, data_obj.datasets.validation,size=1000)
    x_test, y_test = test = _create_dataset(data_obj, data_obj.datasets.test,size=1000)

    f = h5py.File('dataset.hdf5','w')
    dset = f.create_dataset("x_train", data=x_train)
    dset = f.create_dataset("y_train", data=y_train)
    dset = f.create_dataset("x_val", data=x_val)
    dset = f.create_dataset("y_val", data=y_val)
    dset = f.create_dataset("x_test", data=x_test)
    dset = f.create_dataset("y_test", data=y_test)
    return train, validation, test

batch_size = 128
num_classes = 2
epochs = 10
data_augmentation = False 
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_glaucoma_trained_model.h5'

# The data, shuffled and split between train and test sets:
data_obj = get_data()
(x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_data(data_obj)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print("Num positive training examples: ", np.sum(y_train))

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
#model.add(InputLayer(input_tensor=x_train, input_shape=(None,200,200,3)))
model.add(Conv2D(32, (3, 3), padding='same',
                input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

#custom metrics
#def fraction_correct(y_true, y_pred):
#    return K.mean(K.equal(K.greater(y_true,1.5),K.greater(y_pred,1.5)))
def specificity(y_true,y_pred):
    return K.cast(K.all(
        (K.equal(K.argmax(y_true, axis=-1) , 1), K.equal(K.argmax(y_pred,axis=-1), 1))
        ), K.floatx())
def sensitivity(y_true,y_pred):
    return K.cast(K.all(
        (K.equal(K.argmax(y_true, axis=-1) , 0), K.equal(K.argmax(y_pred,axis=-1), 0))
        ), K.floatx())
    #return K.mean(K.all(K.equal(y_true , -1),K.less(y_pred, 0)))

# Let's train the model using RMSprop
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['categorical_accuracy', specificity, sensitivity])

print("Shape is: ",x_train.shape)
print("Label shape: ", y_train.shape) 
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_val, y_val),
              shuffle=True,
              class_weight={0:1,1:24},
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
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        workers=4,
                        class_weight={0:1,1:24},
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
for metric, score in (mode.metrics_names, scores):
    print(metric, ':', score)
exit()
