import tensorflow as tf


def features_to_probs(
    x, output_spec, activation='default', layer_fn=tf.keras.layers.Dense,
    **layer_kwargs):
    if output_spec.shape == ():
        num_classes = 1
    else:
        num_classes = output_spec.shape[-1]
        if hasattr(num_classes, 'value'):
            num_classes = num_classes.value
    if activation == 'default':
        activation = 'sigmoid' if num_classes == 1 else 'softmax'
    layer = layer_fn(
        num_classes, activation=activation, **layer_kwargs)
    probs = layer(x)
    return probs


del tf

__all__ = ['features_to_probs']
