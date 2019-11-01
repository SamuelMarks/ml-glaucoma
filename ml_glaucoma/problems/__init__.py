"""
Problems are mostly adapters operating between base datasets and keras models.

While `tensorflow_datasets.DatasetBuilder`s (see `ml_glaucoma.tfds_builders`
for implementations) provides data download, and serialization and meta-data
collection, problems provide a customizable interface for the models to be
trained. They also include metrics, losses and data augmentation preprocessing.

The `Problem` class provides the general interface, while `TfdsProblem` is a
basic implementation that leverages `tensorflow_datasets.DatasetBuilder`s.
"""

from os import environ

if environ['TF']:
    from ml_glaucoma.problems.tf_keras import *
elif environ['TORCH']:
    from ml_glaucoma.problems.torch import *
else:
    from ml_glaucoma.problems.other import *

from ml_glaucoma.problems.shared import *
