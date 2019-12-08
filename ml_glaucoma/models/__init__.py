"""
Module containing model definitions without compilation.

Each function should have a signature
def get_model(inputs, output_spec, **kwargs) and be annotated with
`@gin.configurable`. The function is responsible for adding all losses except
for those associated with data labels (e.g. regularization losses).
"""

from os import environ

from six import iteritems

from ml_glaucoma.utils.helpers import get_upper_kv

if environ['TF']:
    import tensorflow as tf

    import efficientnet.tfkeras as efficient_net
    from keras_squeeze_excite_network import se_resnet

    from ml_glaucoma.utils import update_d

    valid_models = {
        attr: obj for attr, obj in iteritems(update_d({attr: getattr(tf.keras.applications, attr)
                                                       for attr in get_upper_kv(tf.keras.applications)},
                                                      {attr: getattr(efficient_net, attr)
                                                       for attr in get_upper_kv(efficient_net)},
                                                      **{attr: getattr(se_resnet, attr)
                                                         for attr in get_upper_kv(se_resnet)}))
        if attr not in frozenset(dir(tf.keras.layers) + ['Model']) and not attr.isupper()
    }

elif environ['TORCH']:
    valid_models = {}
else:
    valid_models = {}
