from inspect import currentframe

import gin
import tensorflow as tf

import ml_glaucoma.models.utils.tf_keras
from ml_glaucoma.models import valid_models


@gin.configurable(blacklist=['inputs', 'output_spec'])
def transfer_model(inputs, output_spec, transfer='ResNet50', weights='imagenet',
                   pooling='avg', final_activation='default',
                   kwargs=None):
    if kwargs is None:
        kwargs = {}

    print('transfer_model::transfer:', transfer)
    assert transfer != 'ResNet50'
    features, = valid_models[transfer](
        include_top=False, weights=weights, pooling=pooling,
        input_tensor=inputs, **kwargs).outputs
    probs = ml_glaucoma.models.utils.tf_keras.features_to_probs(
        features, output_spec, activation=final_activation)
    model = tf.keras.models.Model(inputs=inputs, outputs=probs)
    model._name = '_'.join((currentframe().f_code.co_name, transfer))
    return model
