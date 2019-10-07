from __future__ import absolute_import

from os import environ

if environ['TF']:
    from ml_glaucoma.metrics.f1_score.tf_keras import F1Metric
elif environ['TORCH']:
    from ml_glaucoma.metrics.f1_score.torch import F1Metric
else:
    from ml_glaucoma.metrics.f1_score.other import F1Metric
