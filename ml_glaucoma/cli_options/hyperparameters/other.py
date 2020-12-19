from ml_glaucoma.cli_options.base import Configurable

valid_losses = {}
SUPPORTED_LOSSES = tuple(sorted(valid_losses.keys()))

valid_metrics = {}
SUPPORTED_METRICS = tuple(sorted(valid_metrics.keys()))

valid_optimizers = {}
SUPPORTED_OPTIMIZERS = tuple(sorted(valid_optimizers.keys()))


class ConfigurableProblemBase(Configurable):
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
        raise NotImplementedError()


del Configurable

__all__ = [
    "ConfigurableProblemBase",
    "SUPPORTED_LOSSES",
    "SUPPORTED_METRICS",
    "SUPPORTED_OPTIMIZERS",
]
