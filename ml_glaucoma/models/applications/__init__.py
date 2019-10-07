from __future__ import absolute_import

from os import environ

if environ['TF']:
    from ml_glaucoma.models.applications.tf_keras import applications_model
elif environ['TORCH']:
    from ml_glaucoma.models.applications.torch import applications_model
else:
    from ml_glaucoma.models.applications.other import applications_model
