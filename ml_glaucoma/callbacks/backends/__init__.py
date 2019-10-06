from os import environ

if environ['TF']:
    from ml_glaucoma.callbacks.backends.tf_keras import get_callbacks
elif environ['TORCH']:
    from ml_glaucoma.callbacks.backends.torch import get_callbacks
else:
    from ml_glaucoma.callbacks.backends.other import get_callbacks
