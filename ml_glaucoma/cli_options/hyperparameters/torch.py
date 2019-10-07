from ml_glaucoma.cli_options.base import Configurable

valid_losses = {}
SUPPORTED_LOSSES = tuple()

valid_metrics = {}
SUPPORTED_METRICS = tuple()

valid_optimizers = {}
SUPPORTED_OPTIMIZERS = tuple()


class ConfigurableProblemBase(object):
    def build_self(self, builders, map_fn, loss, metrics,
                   precision_thresholds, recall_thresholds,
                   shuffle_buffer, use_inverse_freq_weights,
                   **kwargs):
        raise NotImplementedError()


del Configurable
