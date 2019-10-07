from __future__ import absolute_import

from os import environ

if environ['TF']:
    from ml_glaucoma.losses.kappa_loss.tf_keras import Kappa
elif environ['TORCH']:
    from ml_glaucoma.losses.kappa_loss.torch import Kappa
else:
    from ml_glaucoma.losses.kappa_loss.other import Kappa
