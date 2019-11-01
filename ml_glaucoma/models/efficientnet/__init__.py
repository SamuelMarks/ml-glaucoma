from os import environ

if environ['TF']:
    from ml_glaucoma.models.efficientnet.tf_keras import *
elif environ['TORCH']:
    from ml_glaucoma.models.efficientnet.torch import *
else:
    from ml_glaucoma.models.efficientnet.other import *
