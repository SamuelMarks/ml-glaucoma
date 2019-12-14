from json import dumps

from six import iteritems, iterkeys

from ml_glaucoma.cli_options.base import Configurable
from ml_glaucoma.cli_options.hyperparameters import valid_losses, valid_optimizers
from ml_glaucoma.models import valid_models


class ConfigurableInfo(Configurable):
    description = 'Info subcommand'

    def __init__(self):
        super(ConfigurableInfo, self).__init__()

    def fill_self(self, parser):
        pass

    def build_self(self, rest):
        info = {k: tuple(sorted(iterkeys(v))) for k, v in iteritems({
            'models': valid_models,
            'losses': valid_losses,
            'optimizers': valid_optimizers
        })}
        print(dumps({k: sorted(v) for k, v in iteritems(info)}, sort_keys=True, indent=4))
        return info
