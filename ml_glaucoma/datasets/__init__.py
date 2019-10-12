from __future__ import absolute_import

from os import environ

if environ['TF']:
    from ml_glaucoma.datasets.tfds_builders import *
elif environ['TORCH']:
    from ml_glaucoma.datasets.tfds_builders import *
else:
    from ml_glaucoma.datasets.tfds_builders import *
