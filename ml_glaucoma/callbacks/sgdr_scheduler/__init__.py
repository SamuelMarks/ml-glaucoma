from os import environ

if environ["TF"]:
    from ml_glaucoma.callbacks.sgdr_scheduler.tf_keras import SGDRScheduler
elif environ["TORCH"]:
    from ml_glaucoma.callbacks.sgdr_scheduler.torch import SGDRScheduler
else:
    from ml_glaucoma.callbacks.sgdr_scheduler.other import SGDRScheduler

__all__ = ["SGDRScheduler"]
