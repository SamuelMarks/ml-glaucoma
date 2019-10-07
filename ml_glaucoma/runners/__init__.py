"""Train/validation/predict loops."""
from __future__ import absolute_import

from os import environ

if environ['TF']:
    from ml_glaucoma.runners.tf_keras import *
elif environ['TORCH']:
    from ml_glaucoma.runners.torch import *
else:
    from ml_glaucoma.runners.other import *
