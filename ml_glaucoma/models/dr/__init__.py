from os import environ

if environ['TF']:
    from ml_glaucoma.models.dr.tf_keras import *
elif environ['TORCH']:
    from ml_glaucoma.models.dr.torch import *
else:
    from ml_glaucoma.models.dr.other import *
