"""Train/validation/predict loops."""

from os import environ

if environ["TF"]:
    pass
elif environ["TORCH"]:
    pass
else:
    pass

del environ
