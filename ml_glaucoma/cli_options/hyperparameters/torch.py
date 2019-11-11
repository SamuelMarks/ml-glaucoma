from abc import ABC

from torch import optim as optimizer_module
from torch.nn.modules import loss as losses_module

from ml_glaucoma import problems as p
from ml_glaucoma.aliases import torch2tf_losses, tf2torch_losses
from ml_glaucoma.cli_options.base import Configurable
from ml_glaucoma.utils import pp
from ml_glaucoma.utils.helpers import get_upper_kv

valid_losses = {}
valid_losses.update(get_upper_kv(losses_module))
SUPPORTED_LOSSES = tuple(sorted(valid_losses.keys()))

# TODO: Figure out what metrics are in PyTorch, e.g.: [TensorBoard](https://pytorch.org/docs/stable/tensorboard.html)
# (or maybe just invent them!)
valid_metrics = {}
SUPPORTED_METRICS = tuple(sorted(valid_metrics.keys()))

valid_optimizers = {}
valid_optimizers.update(get_upper_kv(optimizer_module))
SUPPORTED_OPTIMIZERS = tuple(sorted(valid_optimizers.keys()))


class ConfigurableProblemBase(Configurable, ABC):
    def build_self(self, builders, map_fn, loss, metrics,
                   precision_thresholds, recall_thresholds,
                   shuffle_buffer, use_inverse_freq_weights,
                   **kwargs):
        kwargs = dict(
            loss=valid_losses.get(loss, valid_losses[torch2tf_losses.get(loss, tf2torch_losses[loss])]),
            metrics=metrics,
            map_fn=map_fn,
            shuffle_buffer=shuffle_buffer,
            use_inverse_freq_weights=use_inverse_freq_weights)
        if len(builders) == 1:
            return p.TfdsProblem(builder=builders[0], **kwargs)
        print('---' * 10)
        print('ConfigurableProblemBase::build_self::kwargs')
        pp(kwargs)
        return p.TfdsMultiProblem(builders=builders, **kwargs)


del Configurable, optimizer_module, losses_module, get_upper_kv
