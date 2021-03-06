from inspect import currentframe

import gin
import tensorflow as tf
from keras_squeeze_excite_network import se_resnet

import ml_glaucoma.models.utils.tf_keras
from ml_glaucoma.models import utils
from ml_glaucoma.utils.helpers import get_upper_kv

se_resnet_models = get_upper_kv(se_resnet)


@gin.configurable(blacklist=["inputs", "output_spec"])
def se_resnet(
    inputs,
    output_spec,
    application="SEResNet50",
    weights="imagenet",
    pooling="avg",
    final_activation="default",
    kwargs=None,
):
    assert (
        application is not None and application in se_resnet_models
    ), "`application` {!r} not found".format(application)

    if kwargs is None:
        kwargs = {}
    (features,) = se_resnet_models[application](
        include_top=False,
        weights=weights,
        pooling=pooling,
        input_tensor=inputs,
        **kwargs
    ).outputs
    probs = ml_glaucoma.models.utils.tf_keras.features_to_probs(
        features, output_spec, activation=final_activation
    )
    model = tf.keras.models.Model(inputs=inputs, outputs=probs)
    model._name = "_".join((currentframe().f_code.co_name, application))
    return model


del gin, tf, se_resnet, utils, get_upper_kv

__all__ = ["se_resnet"]
