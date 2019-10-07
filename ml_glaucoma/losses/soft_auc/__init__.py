from __future__ import absolute_import

from os import environ

if environ['TF']:
    from ml_glaucoma.losses.soft_auc.tf_keras import SoftAUC
elif environ['TORCH']:
    from ml_glaucoma.losses.soft_auc.torch import SoftAUC
else:
    from ml_glaucoma.losses.soft_auc.other import SoftAUC
