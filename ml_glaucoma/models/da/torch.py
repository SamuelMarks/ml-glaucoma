from inspect import currentframe

import gin


@gin.configurable(blacklist=["inputs", "output_spec"])
def da0(
    inputs,
    output_spec,
    training=None,
    filters=(32, 32, 64),
    dense_units=(64,),
    dropout_rate=0.5,
    conv_activation="relu",
    dense_activation="relu",
    kernel_regularizer=None,
    final_activation="default",
    pooling="flatten",
):
    name = currentframe().f_code.co_name
    raise NotImplementedError(name)


del currentframe, gin

__all__ = ["da0"]
