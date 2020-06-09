from argparse import FileType

from yaml import safe_load as safe_yaml_load

from ml_glaucoma.cli_options.base import Configurable
from ml_glaucoma.cli_options.pipeline.runner import pipeline_runner


class ConfigurablePipeline(Configurable):
    description = 'From a pipeline, get next arguments to try'

    def __init__(self):
        super(ConfigurablePipeline, self).__init__()

    def fill_self(self, parser):
        parser.add_argument(
            '-l', '--logfile', type=FileType('a'), required=True,
            help='logfile to checkpoint whence in the pipeline')
        parser.add_argument(
            '--options', type=safe_yaml_load, required=True,
            help='Object of items to replace argument with, e.g.:'
                 '    { losses:       [ { BinaryCrossentropy: 0 }, { JaccardDistance: 0 } ],'
                 '      models:       [ { DenseNet169: 0 }, { "EfficientNetB0": 0 } ],'
                 '      "optimizers": [ { "Adadelta": 0 }, { "Adagrad": 0 }, { "Adam": 0 } ] }'
        )
        parser.add_argument(
            '--replacement-options', type=safe_yaml_load,
            help='Replacement options, e.g.: {models}'
        )
        parser.add_argument(
            '-k', '--key', required=True,
            help='Start looping through from this key, '
                 'e.g.: `optimizers` will iterate through the optimizer key first, '
                 'then the next key (alphanumerically sorted)'
        )
        parser.add_argument(
            '--threshold', default=1, type=int,
            help='Number of loops to run through. Defaults to 1.'
        )
        parser.add_argument(
            '--dry-run', default=False, action='store_true',
            help='Dry run doesn\'t actually run the subcommand, but prints what it would pass to it'
        )

    def build(self, **kwargs):
        return self.build_self(**kwargs)

    def build_self(self, logfile, key, options, replacement_options, threshold, dry_run, rest):
        return pipeline_runner(logfile, key, options, replacement_options, threshold, dry_run, rest)

    def set_defaults(self, kwargs):
        pass


__all__ = ['ConfigurablePipeline']
