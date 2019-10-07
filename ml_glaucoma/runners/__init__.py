"""Train/validation/predict loops."""
from __future__ import absolute_import

import os

if os.environ['TF']:
    from ml_glaucoma.runners.tf_keras import *
elif os.environ['TORCH']:
    from ml_glaucoma.runners.torch import *
else:
    from ml_glaucoma.runners.other import *
