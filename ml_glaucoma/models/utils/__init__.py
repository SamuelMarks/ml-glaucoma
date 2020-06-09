from os import environ

if environ['TF']:
    from ml_glaucoma.models.utils.tf_keras import *
elif environ['TORCH']:
    from ml_glaucoma.models.utils.torch import *
else:
    from ml_glaucoma.models.utils.other import *

del environ
