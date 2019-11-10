"""
Module containing model definitions without compilation.

Each function should have a signature
def get_model(inputs, output_spec, **kwargs) and be annotated with
`@gin.configurable`. The function is responsible for adding all losses except
for those associated with data labels (e.g. regularization losses).
"""

from os import environ

from six import iterkeys

from ml_glaucoma.utils.helpers import get_upper_kv

if environ['TF']:
    import tensorflow as tf

    import efficientnet.tfkeras as efficient_net
    from keras_squeeze_excite_network import se_resnet

    from ml_glaucoma.models.applications.tf_keras import applications_model
    from ml_glaucoma.utils import update_d

    model_name2model = update_d({attr: getattr(tf.keras.applications, attr)
                                 for attr in get_upper_kv(tf.keras.applications)},
                                {attr: getattr(efficient_net, attr)
                                 for attr in get_upper_kv(efficient_net)},
                                **{attr: getattr(se_resnet, attr)
                                   for attr in get_upper_kv(se_resnet)})

    valid_models = frozenset({
        attr for attr in iterkeys(model_name2model)
        if attr not in frozenset(dir(tf.keras.layers) + ['Model']) and not attr.isupper()
    })

elif environ['TORCH']:
    from ml_glaucoma.models.applications.torch import applications_model

    valid_models = frozenset()
else:
    from ml_glaucoma.models.applications import applications_model

    valid_models = frozenset()
