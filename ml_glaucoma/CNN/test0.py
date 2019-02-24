from __future__ import print_function

import keras
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator


def test0(train_dir, validation_dir, test_dir, pixels):  # type: (str, str, str, int) -> None
    traindatagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
        samplewise_std_normalization=True)

    testdatagen = ImageDataGenerator(
        rescale=1. / 255,
        samplewise_std_normalization=True)

    train_generator = traindatagen.flow_from_directory(
        train_dir,
        target_size=(pixels, pixels),
        batch_size=10,
        color_mode='grayscale')

    validation_generator = testdatagen.flow_from_directory(
        validation_dir,
        target_size=(pixels, pixels),
        batch_size=10,
        color_mode='grayscale')

    model = Sequential()
    model.add(Conv2D(4, (10, 10), input_shape=(pixels, pixels, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.load_weights("third_try.h5")

    opt = keras.optimizers.SGD(lr=0.01, momentum=0.5)
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        epochs=1)

    print(train_generator.class_indices)
    print(validation_generator.class_indices)

    exit(5)
