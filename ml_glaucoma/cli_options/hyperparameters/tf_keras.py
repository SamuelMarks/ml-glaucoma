import tensorflow as tf

from ml_glaucoma import losses as losses_module, metrics as metrics_module
from ml_glaucoma.utils.helpers import get_upper_kv

valid_losses = {loss: getattr(tf.keras.losses, loss)
                for loss in dir(tf.keras.losses)
                if not loss.startswith('_')}
valid_losses.update({loss_name: getattr(tf.losses, loss_name)
                     for loss_name in dir(tf.losses)
                     if not loss_name.startswith('_') and loss_name == 'Reduction'})
valid_losses.update(get_upper_kv(losses_module))
SUPPORTED_LOSSES = tuple(valid_losses.keys())

valid_metrics = {metric: getattr(tf.keras.metrics, metric)
                 for metric in dir(tf.keras.metrics)
                 if not metric.startswith('_') and metric not in frozenset(('serialize', 'deserialize', 'get'))}
valid_metrics.update(get_upper_kv(metrics_module))
SUPPORTED_METRICS = tuple(valid_metrics.keys())

valid_optimizers = get_upper_kv(tf.keras.optimizers)
SUPPORTED_OPTIMIZERS = tuple(valid_optimizers.keys())

# Cleanup namespace
del tf
del losses_module
del metrics_module
del get_upper_kv
