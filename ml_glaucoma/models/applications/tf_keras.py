from inspect import currentframe

import gin
import tensorflow as tf

import ml_glaucoma.models.utils.tf_keras


@gin.configurable(blacklist=['inputs', 'output_spec'])
def applications_model(inputs, output_spec, application='ResNet50',
                       weights='imagenet', pooling='avg', final_activation='default',
                       kwargs=None):
    if kwargs is None:
        kwargs = {}
    features, = getattr(tf.keras.applications, application)(
        include_top=False, weights=weights, pooling=pooling,
        input_tensor=inputs, **kwargs).outputs
    probs = ml_glaucoma.models.utils.tf_keras.features_to_probs(
        features, output_spec, activation=final_activation)
    model = tf.keras.models.Model(inputs=inputs, outputs=probs)
    model._name = '_'.join((currentframe().f_code.co_name, application))
    return model
