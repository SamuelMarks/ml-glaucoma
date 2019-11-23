from argparse import FileType
from datetime import datetime
from functools import partial
from itertools import takewhile
from json import dumps, loads
from os import environ, path, listdir
from sys import modules, stderr

from six import iteritems
from yaml import safe_load as safe_yaml_load

import ml_glaucoma.cli_options.parser
from ml_glaucoma import get_logger
from ml_glaucoma.cli_options.base import Configurable
from ml_glaucoma.utils import update_d, pp

logger = get_logger(modules[__name__].__name__.rpartition('.')[0])


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

    def build(self, **kwargs):
        return self.build_self(**kwargs)

    def build_self(self, logfile, key, options, replacement_options, rest):
        log = lambda obj: logfile.write(
            '{}\n'.format(dumps(update_d({'_dt': datetime.utcnow().isoformat().__str__()}, obj))))

        log({'options': options})
        assert type(options) is dict, '--options value could not be parsed into a Python dictionary, got: {}'.format(
            options
        )
        with open(logfile.name, 'rt') as f:
            prev_logfile_lines = f.readlines()

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
                    for name, value in map(lambda kv: (kv[0], sorted(map(lambda e: tuple(e.items()), kv[1]),
                                                                     key=lambda a: a[0][1])),
                                           filter(lambda kv: not kv[0].startswith('_'), iteritems(options_dict)))}

        options.update(get_sorted_options(options))
        next_key = next(iter(options[key][0].keys()))
        options[key][0][next_key] += 0.5
        options['_next_key'] = next_key

        log({'options': options})

        upsert_rest_arg = partial(ConfigurablePipeline._upsert_cli_arg, cli=rest)
        for k, v in options.items():
            if not k.startswith('_'):
                value = next(iter(v[0].keys()))

                if k == 'models':
                    model = value

                    namespace = ml_glaucoma.cli_options.parser.cli_handler(rest, return_namespace=True)

                    upsert_rest_arg(
                        arg='--model_file',
                        value=path.join(path.dirname(path.dirname(__file__)), 'model_configs', 'transfer.gin')
                    )

                    model_dir = namespace.model_dir

                    optimizer = None if namespace.optimizer == 'Adam' else namespace.optimizer
                    loss = None if namespace.loss == 'BinaryCrossentropy' else namespace.lossj

                    _maybe_suffix = model_dir.rpartition('_')[2]
                    _maybe_suffix = _maybe_suffix if _maybe_suffix.startswith('again') else None

                    _join_with = '_'.join(filter(None, (namespace.dataset[0], model,
                                                        optimizer, loss,
                                                        'epochs', namespace.epochs,
                                                        _maybe_suffix)))
                    model_dir = path.join(path.dirname(model_dir)
                                          if path.isdir(model_dir) and len(listdir(model_dir)) > 0
                                          else model_dir,
                                          _join_with)
                    upsert_rest_arg(
                        arg='--model_dir',
                        value=model_dir
                    )

                    upsert_rest_arg(
                        arg='--model_param',
                        value="application = '{model}'".format(model=model)
                    )
                else:
                    upsert_rest_arg(arg=k, value=value)

        print('-------------------------------------------\n'
              '|                {cmd}ing…                |\n'
              '-------------------------------------------'.format(cmd=rest[0]), sep='')

        if rest[0] != 'train':
            raise NotImplementedError

        err, cli_resp = self._handle_rest(key, next_key, rest, options)
        if err is not None:
            if environ.get('NO_EXCEPTIONS'):
                print(err, file=stderr)
            else:
                raise err
        print('cli_resp:', cli_resp, ';')

        print('-------------------------------------------\n'
              '|            finished {cmd}ing.           |\n'
              '-------------------------------------------'.format(cmd=rest[0]), sep='')

        options[key][0][next_key] += 0.5
        del options['_next_key']
        log(options)

        # TODO: Checkpointing: Check if option was last one tried—by checking if 0.5—and if so, skip to next one

    @staticmethod
    def _upsert_cli_arg(arg, value, cli):
        if not arg.startswith('-'):
            arg = '--{arg}'.format(arg=arg)
        try:
            idx = cli.index(arg)
            cli[idx + 1] = value
        except ValueError:
            cli += [arg, value]
        return cli

    @staticmethod
    def _handle_rest(key, next_key, rest, options):
        assert rest[0] == 'train'

        upsert_rest_arg = partial(ConfigurablePipeline._upsert_cli_arg, cli=rest)

        try:
            model_dir = rest[rest.index('--model_dir') + 1]
        except ValueError:
            model_dir = None

        if model_dir is None:
            logger.warn('model_dir is None, so not incrementing it')
        else:
            try:
                tensorboard_log_dir = rest[rest.index('--tb_log_dir') + 1]
            except ValueError:
                tensorboard_log_dir = model_dir

            reversed_log_dir = model_dir[::-1]
            suffix = int(''.join(takewhile(lambda s: s.isdigit(), reversed_log_dir))[::-1] or 0)
            suffix_s = '{:03d}'.format(suffix)
            if not reversed_log_dir.startswith(reversed_log_dir[:len(suffix_s)] + '_again'[::-1]):
                suffix = 0
                suffix_s = '{:03d}'.format(suffix)
            run = suffix + 1
            print('------------------------\n'
                  '|        RUN {:3d} |\n'
                  '------------------------'.format(run), sep='')
            run_s = '{:03d}'.format(suffix)
            if model_dir.endswith(suffix_s):
                tensorboard_log_dir = ''.join((tensorboard_log_dir[:-len(suffix_s)], run_s))
                model_dir = ''.join((model_dir[:-len(suffix_s)], run_s))
            else:
                tensorboard_log_dir = '{}_again{}'.format(tensorboard_log_dir, suffix_s)
                model_dir = '{}_again{}'.format(model_dir, suffix_s)
            print('New model_dir:'.ljust(16), model_dir)
            # Replace with incremented dirs
            for arg in 'model_dir', 'tensorboard_log_dir':
                upsert_rest_arg(arg=arg, value=locals()[arg])

        # Replace with pipeline argument. Change next line with more complicated—i.e.: more than 1 option change—mods.
        upsert_rest_arg(key, next_key)

        print('Running command:'.ljust(16), '{} {}'.format(
            rest[0], ' '.join(map(lambda r: r if r.startswith('-') else '\'{}\''.format(r),
                                  rest[1:]))))

        err, cli_resp = None, None
        try:
            cli_resp = ml_glaucoma.cli_options.parser.cli_handler(rest)
        except Exception as e:
            err = e

        return err, cli_resp
