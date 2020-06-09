from os import environ

if environ['TF']:
    from ml_glaucoma.losses.smooth_l1.tf_keras import SmoothL1
elif environ['TORCH']:
    from ml_glaucoma.losses.smooth_l1.torch import SmoothL1
else:
    from ml_glaucoma.losses.smooth_l1.other import SmoothL1

del environ
