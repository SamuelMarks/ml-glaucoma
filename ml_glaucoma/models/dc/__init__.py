from os import environ

if environ['TF']:
    from ml_glaucoma.models.dc.tf_keras import *
elif environ['TORCH']:
    from ml_glaucoma.models.dc.torch import *
else:
    from ml_glaucoma.models.dc.other import *
