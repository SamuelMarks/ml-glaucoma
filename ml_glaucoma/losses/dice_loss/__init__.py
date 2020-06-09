from os import environ

if environ['TF']:
    from ml_glaucoma.losses.dice_loss.tf_keras import DiceLoss
elif environ['TORCH']:
    from ml_glaucoma.losses.dice_loss.torch import DiceLoss
else:
    from ml_glaucoma.losses.dice_loss.other import DiceLoss

del environ
