from os import environ

if environ['TF']:
    from ml_glaucoma.callbacks.loading_model_checkpoint.tf_keras import LoadingModelCheckpoint
elif environ['TORCH']:
    from ml_glaucoma.callbacks.loading_model_checkpoint.torch import LoadingModelCheckpoint
else:
    from ml_glaucoma.callbacks.loading_model_checkpoint.other import LoadingModelCheckpoint
