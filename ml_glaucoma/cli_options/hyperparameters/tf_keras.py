from abc import ABC

import tensorflow as tf
import tensorflow_addons as tfa

from ml_glaucoma import losses as losses_module
from ml_glaucoma import metrics as metrics_module
from ml_glaucoma import problems as p
from ml_glaucoma.cli_options.base import Configurable
from ml_glaucoma.utils.helpers import get_upper_kv

valid_losses = {
    loss: getattr(tf.keras.losses, loss)
    for loss in dir(tf.keras.losses)
    if not loss.startswith("_")
    and loss not in frozenset(("Loss", "get", "deserialize"))
}
valid_losses.update(
    {
        loss_name: getattr(tf.losses, loss_name)
        for loss_name in dir(tf.losses)
        if not loss_name.startswith("_") and loss_name == "Reduction"
    }
)
valid_losses.update(get_upper_kv(losses_module))

SUPPORTED_LOSSES = tuple(sorted(valid_losses.keys()))

valid_metrics = {
    metric: getattr(tf.keras.metrics, metric)
    for metric in dir(tf.keras.metrics)
    if not metric.startswith("_")
    and metric not in frozenset(("serialize", "deserialize", "get"))
}
valid_metrics.update(get_upper_kv(metrics_module))
SUPPORTED_METRICS = tuple(sorted(valid_metrics.keys()))

valid_optimizers = {
    optimizer_name: optimizer
    for optimizer_name, optimizer in get_upper_kv(tf.keras.optimizers).items()
    if optimizer_name not in ("Optimizer", "OptimizerV2")
}
SUPPORTED_OPTIMIZERS = tuple(sorted(valid_optimizers.keys()))


class ConfigurableProblemBase(Configurable, ABC):
    def build_self(
        self,
        builders,
        map_fn,
        loss,
        metrics,
        precision_thresholds,
        recall_thresholds,
        shuffle_buffer,
        use_inverse_freq_weights,
        **kwargs
    ):
        metrics = [
            (
                tf.keras.metrics.deserialize(dict(class_name=metric, config={}))
                if metric in dir(tf.keras.metrics)
                else valid_metrics[metric]
            )
            for metric in metrics
        ]

        # multiple threshold values don't seem to work for metrics
        # metrics_module.F1All(
        #    name='F1ALL',
        #    num_classes=2,
        #    average='micro',
        #    writer=tf.summary.create_file_writer(kwargs['tensorboard_log_dir'],
        #                                         filename_suffix='.metrics')
        # ),

        metrics += (
            [
                tf.metrics.AUC(curve="PR", name="AUCPR"),
                tfa.metrics.F1Score(num_classes=2, average="micro"),
                # tfa.metrics.CohenKappa(num_classes=2),
                # tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2),
                tfa.metrics.FBetaScore(num_classes=2, average="micro"),
            ]
            + [
                tf.keras.metrics.TruePositives(
                    thresholds=[t], name="tp{:d}".format(int(100 * t))
                )
                for t in precision_thresholds
            ]
            + [
                tf.keras.metrics.FalsePositives(
                    thresholds=[t], name="fp{:d}".format(int(100 * t))
                )
                for t in precision_thresholds
            ]
            + [
                tf.keras.metrics.TrueNegatives(
                    thresholds=[t], name="tn{:d}".format(int(100 * t))
                )
                for t in precision_thresholds
            ]
            + [
                tf.keras.metrics.FalseNegatives(
                    thresholds=[t], name="fn{:d}".format(int(100 * t))
                )
                for t in precision_thresholds
            ]
            + [
                tf.keras.metrics.Precision(
                    thresholds=[t], name="precision{:d}".format(int(100 * t))
                )
                for t in precision_thresholds
            ]
            + [
                tf.keras.metrics.Recall(
                    thresholds=[r], name="recall{:d}".format(int(100 * r))
                )
                for r in recall_thresholds
            ]
        )

        kwargs = dict(
            loss=(
                tf.keras.losses.deserialize(dict(class_name=loss, config={}))
                if loss in dir(tf.keras.losses)
                else valid_losses[loss]
            ),
            metrics=metrics,
            map_fn=map_fn,
            shuffle_buffer=shuffle_buffer,
            use_inverse_freq_weights=use_inverse_freq_weights,
        )
        if len(builders) == 1:
            return p.TfdsProblem(builder=builders[0], **kwargs)
        return p.TfdsMultiProblem(builders=builders, **kwargs)


# Cleanup namespace
del losses_module, get_upper_kv

__all__ = [
    "ConfigurableProblemBase",
    "valid_losses",
    "SUPPORTED_LOSSES",
    "valid_metrics",
    "SUPPORTED_METRICS",
    "valid_optimizers",
    "SUPPORTED_OPTIMIZERS",
]
