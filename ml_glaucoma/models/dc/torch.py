from inspect import currentframe

import gin


@gin.configurable(blacklist=["inputs", "output_spec"])
def dc0(
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


@gin.configurable(blacklist=["inputs", "output_spec"])
def dc1(
    inputs,
    output_spec,
    training=None,
    dropout_rate=0.5,
    num_dropout_layers=4,
    kernel_regularizer=None,
    conv_activation="relu",
    final_activation="default",
    pooling="avg",
):
    name = currentframe().f_code.co_name
    raise NotImplementedError(name)


@gin.configurable(blacklist=["inputs", "output_spec"])
def dc2(
    inputs,
    output_spec,
    training=None,
    filters=(32, 32, 64),
    dense_units=(32,),
    dropout_rate=0.5,
    conv_activation="relu",
    dense_activation="relu",
    kernel_regularizer=None,
    final_activation="default",
    pooling="flatten",
):
    name = currentframe().f_code.co_name
    raise NotImplementedError(name)


@gin.configurable(blacklist=["inputs", "output_spec"])
def dc3(inputs, output_spec, final_activation="default", class_mode="binary"):
    name = currentframe().f_code.co_name
    raise NotImplementedError(name)


@gin.configurable(blacklist=["inputs", "output_spec"])
def dc4(
    inputs, output_spec, final_activation="default", class_mode="binary", dropout=4
):
    name = currentframe().f_code.co_name
    raise NotImplementedError(name)


del currentframe, gin

__all__ = ["dc0", "dc1", "dc2", "dc3", "dc4"]
