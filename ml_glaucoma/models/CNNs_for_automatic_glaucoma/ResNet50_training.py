import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from datetime import datetime
import os

#Parameters
IMAGE_SIZE = 224                              # Image size (224x224)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
LEARNING_RATE_SCHEDULE_FACTOR = 0.1           # Parameter used for reducing learning rate
LEARNING_RATE_SCHEDULE_PATIENCE = 5           # Parameter used for reducing learning rate
MAX_EPOCHS = 10                               # Maximum number of training epochs
num_classes=2
training_path="../../../../mnt-lg/train"
val_path="../../../../mnt-lg/valid"
test_path="../../../../mnt-lg/test"
ds_dir = '/tmp/tf0'

#Load the data
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                             featurewise_center=True,
                                                             featurewise_std_normalization=True)

train_gen = train_datagen.flow_from_directory(training_path,
                                        class_mode="categorical",
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=BATCH_SIZE)

val_gen = train_datagen.flow_from_directory(directory=val_path,
                                        class_mode="categorical",
                                        shuffle=False,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=BATCH_SIZE)

test_gen = train_datagen.flow_from_directory(directory=test_path,
                                             class_mode=None,
                                             shuffle=False,
                                             target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                             batch_size=BATCH_SIZE)


#Define F1 score
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#Define callbacks
logdir = os.path.join(os.path.dirname(ds_dir), 'ml-logs' + datetime.now().strftime("%Y%m%d-%H%M%S")+' ResNet50')

mcp = tf.keras.callbacks.ModelCheckpoint("resnet50.h5", monitor="val_f1", save_best_only=True, save_weights_only=True, verbose=1,mode='max')
rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1', factor=LEARNING_RATE_SCHEDULE_FACTOR, mode='max', patience=LEARNING_RATE_SCHEDULE_PATIENCE, min_lr=1e-8, verbose=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
callbacks = [mcp, rlr, tensorboard_callback]

#Define the model
def get_model(IMAGE_SIZE, num_classes):
    base_model = keras.applications.ResNet50(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),
                                        include_top=False, #Set to false to train
                                        weights='imagenet')
    base_model.trainable = True

    model = keras.Sequential([
      base_model,
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Dense(num_classes, activation='sigmoid')
    ])

    # Print out model summary
    #model.summary()
    #base_model.summary()
    return model


#Training
device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'

with tf.device(device):
    steps_per_epoch = train_gen.n // BATCH_SIZE
    validation_steps = val_gen.n // BATCH_SIZE
    model=get_model(IMAGE_SIZE,num_classes)
    model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE), loss='binary_crossentropy', metrics=[f1])

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=MAX_EPOCHS,
                                  verbose=1,
                                  validation_data=val_gen,
                                  validation_steps=validation_steps,
                                  callbacks=callbacks)

