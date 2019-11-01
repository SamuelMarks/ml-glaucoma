from os import environ

if environ['TF']:
    from ml_glaucoma.models.squeeze_excite_resnet.tf_keras import *
elif environ['TORCH']:
    from ml_glaucoma.models.squeeze_excite_resnet.torch import *
else:
    from ml_glaucoma.models.squeeze_excite_resnet.other import *
