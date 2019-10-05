from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from inspect import currentframe

import efficientnet.tfkeras as efficientnet_tfkeras_models
import gin
import tensorflow as tf

from ml_glaucoma.models import util
from ml_glaucoma.utils.helpers import get_upper_kv

efficientnet_models = get_upper_kv(efficientnet_tfkeras_models)


@gin.configurable(blacklist=['inputs', 'output_spec'])
def efficient_net(inputs, output_spec, application='EfficientNetB0',
                  weights='imagenet', pooling='avg', final_activation='default',
                  kwargs=None):
    assert application is not None and application in efficientnet_models, '`application` not found'

    if kwargs is None:
        kwargs = {}
    features, = efficientnet_models[application](
        include_top=False, weights=weights, pooling=pooling,
        input_tensor=inputs, **kwargs).outputs
    probs = util.features_to_probs(
        features, output_spec, activation=final_activation)
    model = tf.keras.models.Model(inputs=inputs, outputs=probs)
    model._name = '_'.join((currentframe().f_code.co_name, application))
    return model
