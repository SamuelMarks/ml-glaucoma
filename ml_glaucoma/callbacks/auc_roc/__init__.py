from __future__ import absolute_import

from os import environ

if environ['TF']:
    from ml_glaucoma.callbacks.auc_roc.tf_keras import AucRocCallback
elif environ['TORCH']:
    from ml_glaucoma.callbacks.auc_roc.torch import AucRocCallback
else:
    from ml_glaucoma.callbacks.auc_roc.other import AucRocCallback
