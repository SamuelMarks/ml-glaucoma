from os import environ

if environ['TF']:
    from ml_glaucoma.models.da.tf_keras import *
elif environ['TORCH']:
    from ml_glaucoma.models.da.torch import *
else:
    from ml_glaucoma.models.da.other import *

del environ
