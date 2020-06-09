from os import environ

if environ['TF']:
    from ml_glaucoma.datasets.tfds_builders import *
elif environ['TORCH']:
    from ml_glaucoma.datasets.torch import *
else:
    from ml_glaucoma.datasets.tfds_builders import *

del environ
