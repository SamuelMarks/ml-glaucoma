from os import environ

if environ['TF']:
    from ml_glaucoma.metrics.f1_all.tf_keras import F1All
elif environ['TORCH']:
    from ml_glaucoma.metrics.f1_all.torch import F1All
else:
    from ml_glaucoma.metrics.f1_all.other import F1All

del environ
