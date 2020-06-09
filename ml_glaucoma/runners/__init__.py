"""Train/validation/predict loops."""

from os import environ

if environ['TF']:
    from ml_glaucoma.runners.tf_keras import *
elif environ['TORCH']:
    from ml_glaucoma.runners.torch import *
else:
    from ml_glaucoma.runners.other import *

del environ
