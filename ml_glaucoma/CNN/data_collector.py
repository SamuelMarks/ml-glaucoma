# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import h5py
from PIL import Image
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import os

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile
import pandas as pd

# CVDF mirror of http://yann.lecun.com/exdb/mnist/
DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(folder):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].

  """

    if(os.path.isfile("dataset.hdf5")):
        f = h5py.File("dataset.hdf5",'r')
        Xdset = f.get('X')
        ydset = f.get('y')
        X = numpy.array(Xdset)
        y = numpy.array(ydset)
        return X,y
 
    print("Reading xls files...")
    files = [
          'Annotation_Base11.xls',
          'Annotation_Base12.xls',
          'Annotation_Base13.xls',
          'Annotation_Base14.xls',
          'Annotation_Base21.xls',
          'Annotation_Base22.xls',
          'Annotation_Base23.xls',
          'Annotation_Base24.xls',
          'Annotation_Base31.xls',
          'Annotation_Base32.xls',
          'Annotation_Base33.xls',
          'Annotation_Base34.xls',
          ]
    images = []
    retinopathy_grade = []
    risk_macular_edema = []
    for file in files:
        x = pd.ExcelFile(file)
        df = x.parse('Feuil1')
        images += list(df['Image name'])
        retinopathy_grade += list(df['Retinopathy grade'])
        risk_macular_edema += list(df['Risk of macular edema '])#currently unused

    data = []
    print("Loading data from files...")
    for image in images:
        print(image, end='\r')
        with Image.open(image) as im:
            im = im.resize((300,300))
            data += [numpy.array(im)]
    data = numpy.stack(data)
    print()
    print(data.shape)
    X = data
    y = retinopathy_grade
    from sklearn.utils import shuffle
    X, y = shuffle(X,y,random_state=0)
    print(X.shape,len(y))
    f = h5py.File('dataset.hdf5','w')
    dset = f.create_dataset("X", data=X)
    dset = f.create_dataset("y", data=y)

    return X, y


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot



class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=False,
                   test_size=100,
                   validation_size=100,
                   seed=None,
                   source_url=DEFAULT_SOURCE_URL):

    X, y = extract_images(train_dir)
    train_images = X
    train_labels = y

    if one_hot:
        train_labels = dense_to_one_hot(train_labels, 10)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    # test_images = train_images[:test_size]
    # test_labels = train_labels[:test_size]
    # train_images = train_images[test_size:]
    # train_labels = train_labels[test_size:]



    # options = dict(dtype=dtype, reshape=reshape, seed=seed)

    # train = DataSet(train_images, train_labels, **options)
    # validation = DataSet(validation_images, validation_labels, **options)
    # test = DataSet(test_images, test_labels, **options)

    # return base.Datasets(train=train, validation=validation, test=test)
    return (train_images,train_labels),(validation_images,validation_labels)


def load_messidor(test_size=75, validation_size=75,one_hot=False):
  return read_data_sets('./', 
          test_size=test_size, 
          validation_size=validation_size,
          one_hot=one_hot,
          dtype=dtypes.uint8)

if __name__ == '__main__':

    (_, y), (_, y_val) = read_data_sets('./')
    print(y)
    print(numpy.mean(y > 1.5))
    print(numpy.mean(y_val > 1.5))
