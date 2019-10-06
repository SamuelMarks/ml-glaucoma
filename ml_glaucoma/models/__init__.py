"""
Module containing model definitions without compilation.

Each function should have a signature
def get_model(inputs, output_spec, **kwargs) and be annotated with
`@gin.configurable`. The function is responsible for adding all losses except
for those associated with data labels (e.g. regularization losses).
"""
from os import environ

if environ['TF']:
    from ml_glaucoma.models.applications.tf_keras import applications_model
elif environ['TORCH']:
    from ml_glaucoma.models.applications.torch import applications_model
else:
    from ml_glaucoma.models.applications import applications_model
