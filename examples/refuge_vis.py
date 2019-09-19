from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow_datasets as tfds
from ml_glaucoma.tfds_builders.refuge import Refuge
import matplotlib.pyplot as plt
import functools
from ml_glaucoma.problems import preprocess_example

builder = Refuge()
builder.download_and_prepare()
dataset = builder.as_dataset(as_supervised=True, split='validation')

dataset = dataset.map(functools.partial(
    preprocess_example, pad_to_square=True, resolution=(256, 256)))


for image, label in tfds.as_numpy(dataset):
    image -= np.min(image)
    image /= np.max(image)
    plt.imshow(image)
    print('label: {}'.format(label))
    plt.show()