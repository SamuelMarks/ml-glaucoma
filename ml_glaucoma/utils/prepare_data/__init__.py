from __future__ import absolute_import

from os import environ

if environ['TF']:
    from ml_glaucoma.utils.prepare_data.tf_keras import *
elif environ['TORCH']:
    from ml_glaucoma.utils.prepare_data.torch import *
else:
    from ml_glaucoma.utils.prepare_data.other import *
