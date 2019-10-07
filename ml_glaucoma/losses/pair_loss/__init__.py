from __future__ import absolute_import

from os import environ

if environ['TF']:
    from ml_glaucoma.losses.pair_loss.tf_keras import PairLoss
elif environ['TORCH']:
    from ml_glaucoma.losses.pair_loss.torch import PairLoss
else:
    from ml_glaucoma.losses.pair_loss.other import PairLoss
