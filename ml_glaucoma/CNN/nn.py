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

DATA_SAVE_LOCATION = '/mnt/datasets/400x400balanced_dataset.hdf5'

def prepare_data():
    def _parse_function(filename):
        image = cv2.imread(filename)
        image_resized = cv2.resize(image, (400,400))
        global i
        print("Importing image ", i, end='\r')
        i += 1
        return image_resized
    def _get_filenames(neg_ids,pos_ids,id_to_imgs):
        #returns filenames list and labels list
        labels = []
        filenames = []
        for id in list(pos_ids)+list(neg_ids[:120]):
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

        print("Total images: ", len(img_names))

        global i
        i = 1
        dataset_tensor = np.stack(list(map(_parse_function,img_names)))
        print()

        return dataset_tensor, data_labels

    if(os.path.isfile(DATA_SAVE_LOCATION)):
        f = h5py.File(DATA_SAVE_LOCATION,'r')
        x_train_dset = f.get('x_train')
        y_train_dset = f.get('y_train')
        x_test_dset = f.get('x_test')
        y_test_dset = f.get('y_test')
        # X = numpy.array(Xdset)
        return (x_train_dset,y_train_dset),(x_test_dset,y_test_dset)

    data_obj = get_data()
    x, y = _create_dataset(data_obj)

    x, y = shuffle(x,y,random_state=0)
    x = x.astype('float32')
    x /= 255.

    train_fraction = 0.9
    train_amount = int(x.shape[0]*0.9)
    x_train, y_train = x[:train_amount],y[:train_amount]
    x_test, y_test = x[train_amount:],y[train_amount:]

    f = h5py.File(DATA_SAVE_LOCATION,'w')
    x_train = f.create_dataset("x_train", data=x_train, )#compression='lzf')
    y_train = f.create_dataset("y_train", data=y_train,)# compression='lzf')
    x_test = f.create_dataset("x_test", data=x_test, )#compression='lzf')
    y_test = f.create_dataset("y_test", data=y_test, )#compression='lzf')

    return (x_train, y_train),(x_test, y_test)

batch_size = 256
num_classes = 2
epochs = 10
data_augmentation = False
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_glaucoma_trained_model.h5'
CIFAR = True
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

# Convert class vectors to binary class matrices.
if categorical:
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
#model.add(InputLayer(input_tensor=x_train, input_shape=(None,200,200,3)))
model.add(Conv2D(32, (3, 3), padding='same',
                input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
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

model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.8))
if categorical:
    model.add(Dense(2))
    model.add(Activation('softmax'))
else:
    model.add(Dense(1))

#custom metrics for categorical
def specificity(y_true,y_pred):
    return K.cast(K.all(
        (K.equal(K.argmax(y_true, axis=-1) , 1), K.equal(K.argmax(y_pred,axis=-1), 1))
        ,axis=0), K.floatx())
def sensitivity(y_true,y_pred):
    return K.cast(K.all(
        (K.equal(K.argmax(y_true, axis=-1) , 2), K.equal(K.argmax(y_pred,axis=-1), 2))
        ,axis=0), K.floatx())

#custom metrics for single output
def spec(y_true, y_pred):
    return K.cast(K.all(
        (K.greater(y_true, 0), K.greater(y_pred, 0))
        ,axis=0), K.floatx())
def sens(y_true, y_pred):
    return K.cast(K.all(
        (K.less(y_true, 0), K.less(y_pred, 0))
        ,axis=0), K.floatx())

#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#opt = keras.optimizers.RMSprop()
opt = keras.optimizers.SGD(lr=0.0001, decay = 1e-6, momentum=0.9)
if categorical:
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=[specificity, sensitivity])
else:
    model.compile(loss='mse',
              optimizer=opt,
              metrics=[spec, sens])

print("Shape is: ",x_train.shape)
print("Label shape: ", y_train.shape) 

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.09,
              #shuffle='batch',
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
for metric, score in zip(model.metrics_names, scores):
    print(metric, ':', score)

with open('log.txt', 'a') as f:
    print("\nTest results\n",file=f)
    for metric, score in zip(model.metrics_names, scores):
        print(metric, ':', score, file=f)
    
results = model.predict(x_test)
print(results)
print(np.mean(results,axis=0))

exit()    
