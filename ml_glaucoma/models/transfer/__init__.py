from os import environ

if environ['TF']:
    from ml_glaucoma.models.transfer.tf_keras import transfer_model
elif environ['TORCH']:
    from ml_glaucoma.models.transfer.torch import applications_model
else:
    from ml_glaucoma.models.transfer.other import applications_model
