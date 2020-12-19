from os import environ

if environ["TF"]:
    pass
elif environ["TORCH"]:
    raise NotImplementedError()
else:
    raise NotImplementedError()

del environ
