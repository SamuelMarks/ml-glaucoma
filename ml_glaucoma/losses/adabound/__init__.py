from os import environ

if environ['TF']:
    from ml_glaucoma.losses.adabound.tf_keras import AdaBound
elif environ['TORCH']:
    from ml_glaucoma.losses.adabound.torch import AdaBound
else:
    from ml_glaucoma.losses.adabound.other import AdaBound
