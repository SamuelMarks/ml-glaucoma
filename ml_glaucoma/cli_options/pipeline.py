from argparse import FileType
from datetime import datetime

from six import iteritems
from yaml import load as yaml_load

from ml_glaucoma.cli_options.base import Configurable
from ml_glaucoma.runners import train


class ConfigurablePipeline(Configurable):
    description = 'Pipeline for ML'

    def __init__(self):
        super(ConfigurablePipeline, self).__init__()

    def fill_self(self, parser):
        parser.add_argument(
            '-l', '--logfile', type=FileType('a'), required=True,
            help='logfile to checkpoint whence in the pipeline')
        parser.add_argument(
            '--options', type=yaml_load, required=True,
            help='Object of items to replace argument with, e.g.: '
                 '{ losses: [BinaryCrossentropy, JaccardDistance], '
                 ' models: [DenseNet169, "EfficientNetB0"], '
                 ' "optimizers": ["Adadelta", "Adagrad", "Adam"] }'
        )

    def build_self(self, logfile, options):
        logfile.write('{dt}\t{msg}'.format(dt=datetime.now(), msg='pipeline::build_self'))
        assert options is dict

        # TODO: Check if option was last one tried, and if so, skip to next one
        for k, v in iteritems(options):
            logfile.write('{dt}\t{option}'.format(dt=datetime.now(), option=k))
            train(**{k: v})  # TODO: Add default args, figure out how to get same args as `train` from `argv`
