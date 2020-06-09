from os import environ

if environ['TF']:
    from ml_glaucoma.metrics.auc_all.tf_keras import AUCall
elif environ['TORCH']:
    from ml_glaucoma.metrics.auc_all.torch import AUCall
else:
    from ml_glaucoma.metrics.auc_all.other import AUCall

del environ
