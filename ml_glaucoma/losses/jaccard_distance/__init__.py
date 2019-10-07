from __future__ import absolute_import

from os import environ

if environ['TF']:
    from ml_glaucoma.losses.jaccard_distance.tf_keras import JaccardDistance
elif environ['TORCH']:
    from ml_glaucoma.losses.jaccard_distance.torch import JaccardDistance
else:
    from ml_glaucoma.losses.jaccard_distance.other import JaccardDistance
