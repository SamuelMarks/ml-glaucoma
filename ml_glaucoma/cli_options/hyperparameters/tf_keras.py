import tensorflow as tf

from ml_glaucoma import losses as losses_module, metrics as metrics_module, problems as p
from ml_glaucoma.cli_options.base import Configurable
from ml_glaucoma.metrics import F1Metric
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


class ConfigurableProblemBase(Configurable):
    def build_self(self, builders, map_fn, loss, metrics,
                   precision_thresholds, recall_thresholds,
                   shuffle_buffer, use_inverse_freq_weights,
                   **kwargs):
        metrics = [
            (tf.keras.metrics.deserialize(dict(class_name=metric, config={})) if metric in dir(tf.keras.metrics)
             else valid_metrics[metric])
            for metric in metrics
        ]

        # multiple threshold values don't seem to work for metrics
        metrics += [
                       tf.keras.metrics.TruePositives(
                           thresholds=[t],
                           name='tp{:d}'.format(int(100 * t)))
                       for t in precision_thresholds
                   ] + [
                       tf.keras.metrics.FalsePositives(
                           thresholds=[t],
                           name='fp{:d}'.format(int(100 * t)))
                       for t in precision_thresholds
                   ] + [
                       tf.keras.metrics.TrueNegatives(
                           thresholds=[t],
                           name='tn{:d}'.format(int(100 * t)))
                       for t in precision_thresholds
                   ] + [
                       tf.keras.metrics.FalseNegatives(
                           thresholds=[t],
                           name='fn{:d}'.format(int(100 * t)))
                       for t in precision_thresholds
                   ] + [
                       F1Metric(
                           num_classes=2,
                           threshold=t,
                           name='f1{:d}'.format(int(100 * t)))
                       for t in precision_thresholds
                   ] + [
                       tf.keras.metrics.Precision(
                           thresholds=[t],
                           name='precision{:d}'.format(int(100 * t)))
                       for t in precision_thresholds
                   ] + [
                       tf.keras.metrics.Recall(
                           thresholds=[r],
                           name='recall{:d}'.format(int(100 * r)))
                       for r in recall_thresholds]

        kwargs = dict(
            loss=(tf.keras.losses.deserialize(dict(class_name=loss, config={})) if loss in dir(tf.keras.losses)
                  else valid_losses[loss]),
            metrics=metrics,
            map_fn=map_fn,
            shuffle_buffer=shuffle_buffer,
            use_inverse_freq_weights=use_inverse_freq_weights)
        if len(builders) == 1:
            return p.TfdsProblem(builder=builders[0], **kwargs)
        return p.TfdsMultiProblem(builders=builders, **kwargs)


# Cleanup namespace
for obj in losses_module, metrics_module, get_upper_kv:
    del obj
