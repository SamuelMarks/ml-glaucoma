from argparse import FileType
from datetime import datetime
from json import dumps, loads

from six import iteritems
from yaml import safe_load as safe_yaml_load

from ml_glaucoma.cli_options.base import Configurable
from ml_glaucoma.utils import update_d


class ConfigurablePipeline(Configurable):
    description = 'From a pipeline, get next arguments to try'

    def __init__(self):
        super(ConfigurablePipeline, self).__init__()

    def fill_self(self, parser):
        parser.add_argument(
            '-l', '--logfile', type=FileType('a'), required=True,
            help='logfile to checkpoint whence in the pipeline')
        parser.add_argument(
            '-c', '--cmd', required=True,
            help='Command to run, e.g.: `train`')
        parser.add_argument(
            '--options', type=safe_yaml_load, required=True,
            help='Object of items to replace argument with, e.g.:'
                 '    { losses:       [ { BinaryCrossentropy: 0 }, { JaccardDistance: 0 } ],'
                 '      models:       [ { DenseNet169: 0 }, { "EfficientNetB0": 0 } ],'
                 '      "optimizers": [ { "Adadelta": 0 }, { "Adagrad": 0 }, { "Adam": 0 } ] }'
        )
        parser.add_argument(
            '-k', '--key',
            help='Start looping through from this key, '
                 'e.g.: `optimizers` will iterate through the optimizer key first, '
                 'then the next key (alphanumerically sorted)'
        )

    def build(self, **kwargs):
        return self.build_self(**kwargs)

    def build_self(self, logfile, cmd, key, options):
        log = lambda obj: logfile.write(
            '{}\n'.format(dumps(update_d({'_dt': datetime.utcnow().isoformat().__str__()}, obj))))

        log({'options': options})
        assert type(options) is dict, '--options value could not be parsed into a Python dictionary, got: {}'.format(
            options
        )
        with open(logfile.name, 'rt') as f:
            prev_logfile_lines = f.readlines()

        if cmd != 'train':
            raise NotImplementedError

        train_kwargs = options.pop('train_kwargs', {})

        get_shape = lambda obj: {
            k: list(map(lambda o: next(iter(o.keys())), v)) if type(v) is list and type(v[0]) is dict else v
            for k, v in iteritems(obj)
            if not k.startswith('_')
        }

        incoming_shape = get_shape(options)
        last_line = None
        if len(prev_logfile_lines):
            last_options_line = last_line = loads(prev_logfile_lines[-1])
            last_shape = get_shape(last_line)
            i = -2
            while (len(last_line) != len(last_shape) and last_shape != incoming_shape and i != -len(prev_logfile_lines)
                   and len(prev_logfile_lines) > 1):
                last_options_line = loads(prev_logfile_lines[i])
                last_shape = get_shape(last_options_line)
                i -= 1

            if last_shape == incoming_shape:
                incoming_shape = last_shape
                options.update(last_options_line)

        def get_sorted_options(options_dict):
            return {name: list(map(dict, value))
                    for name, value in map(lambda kv: (kv[0], sorted(map(lambda e: tuple(e.items()),
                                                                         kv[1]), key=lambda a: a[0][1])),
                                           filter(lambda kv: not kv[0].startswith('_'), iteritems(options_dict)))}

        options = get_sorted_options(options)
        next_key = next(iter(options[key][0].keys()))
        options[key][0][next_key] += 0.5
        options['_next_key'] = next_key

        log(options)

        print('training...')

        options[key][0][next_key] += 0.5
        del options['_next_key']
        log(options)

        # TODO: Check if option was last one tried, and if so, skip to next one
        '''
        for k, v in iteritems(options):
            logfile.writeline(dumps(dict(dt=datetime.now(), option=k)))
            train(**{k: v})  # TODO: Add default args, figure out how to get same args as `train` from `argv`
        '''
