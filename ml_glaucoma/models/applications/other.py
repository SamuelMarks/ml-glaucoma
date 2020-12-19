from inspect import currentframe

import gin


@gin.configurable(blacklist=["inputs", "output_spec"])
def applications_model(
    inputs,
    output_spec,
    application="ResNet50",
    weights="imagenet",
    pooling="avg",
    final_activation="default",
    kwargs=None,
):
    name = "_".join((currentframe().f_code.co_name, application))
    raise NotImplementedError(name)


__all__ = ["applications_model"]
