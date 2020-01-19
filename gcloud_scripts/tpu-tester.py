# Taken from https://github.com/tensorflow/docs/blob/bef6f89/site/en/guide/tpu.ipynb

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import tensorflow_datasets as tfds

print('''os.environ['COLAB_TPU_ADDR']:''', os.environ['COLAB_TPU_ADDR'], ';')

assert 'COLAB_TPU_ADDR' in os.environ
assert len(os.environ['COLAB_TPU_ADDR'])

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

def create_model():
  return tf.keras.Sequential(
      [tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(10)])

def get_dataset(batch_size=200):
  datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True,
                             try_gcs=True)
  mnist_train, mnist_test = datasets['train'], datasets['test']

  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.0

    return image, label

  train_dataset = mnist_train.map(scale).shuffle(10000).batch(batch_size)
  test_dataset = mnist_test.map(scale).batch(batch_size)

  return train_dataset, test_dataset

strategy = tf.distribute.experimental.TPUStrategy(resolver)
with strategy.scope():
  model = create_model()
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])

train_dataset, test_dataset = get_dataset()

model.fit(train_dataset,
          epochs=5,
          validation_data=test_dataset)
