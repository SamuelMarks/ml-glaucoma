from os import environ

if environ['TF']:
    from ml_glaucoma.models.transfer.tf_keras import transfer_model
elif environ['TORCH']:
    from ml_glaucoma.models.transfer.torch import transfer_model
else:
    from ml_glaucoma.models.transfer.other import transfer_model
