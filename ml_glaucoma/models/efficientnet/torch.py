from inspect import currentframe

import gin


@gin.configurable(blacklist=["inputs", "output_spec"])
def efficient_net(
    inputs,
    output_spec,
    application="EfficientNetB0",
    weights="imagenet",
    pooling="avg",
    final_activation="default",
    kwargs=None,
):
    name = "_".join((currentframe().f_code.co_name, application))
    raise NotImplementedError(name)


del currentframe, gin

__all__ = ["efficient_net"]
