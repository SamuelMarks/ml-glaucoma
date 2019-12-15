from datetime import datetime
from functools import partial
from itertools import takewhile
from json import dumps, loads
from os import environ, path, listdir
from shutil import copyfile
from sys import stderr, modules
from tempfile import gettempdir, mkdtemp

from six import iteritems

import ml_glaucoma.cli_options.parser
from ml_glaucoma import get_logger
from ml_glaucoma.utils import update_d
from ml_glaucoma.cli_options.logparser.utils import get_metrics, parse_line
import ml_glaucoma.cli_options.parser
from ml_glaucoma.cli_options.hyperparameters import SUPPORTED_LOSSES, SUPPORTED_OPTIMIZERS
from ml_glaucoma.models import valid_models

logger = get_logger(modules[__name__].__name__.rpartition('.')[0])


def pipeline_runner(logfile, key, options, replacement_options, threshold, dry_run, rest):
    log = lambda obj: logfile.write(
        '{}\n'.format(dumps(update_d({'_dt': datetime.utcnow().isoformat().__str__()}, obj))))

    log({'options': options})
    actual_run, final_result = 0, None
    while actual_run < threshold:
        actual_run += 1
        next_options = _new_prepare_options(log, logfile, options, rest)
        _new_modify_options(next_options, rest)

        final_result = _execute_command(key, log, None, options, dry_run, rest)

    logfile.close()
    return final_result

    # TODO: Checkpointing: Check if option was last one tried—by checking if 0.5—and if so, skip to next one

def _new_prepare_options(log, logfile, options, rest, try_all=True):
    '''
        This is run for every experiment
        From rest, we can use get_metrics to open 'tensorboard_log_dir' to get the result of all the previous experiments
            (change the prefix to 'epoch_') (using the validation subdirectory)
            if path is empty just return
        Change what is returned and
        Build some data structure to determine next best choice
        Select set of options to run for next experiment
        Call upsert_rest_arg() to set parameters for the next experiment
    '''
    rest_namespace = ml_glaucoma.cli_options.parser.cli_handler(cmd=rest, return_namespace=True)
    # if not path.isdir(rest_namespace.tensorboard_log_dir) or len(listdir(rest_namespace.tensorboard_log_dir)) == 0:
    #     return  # first run!

    print('rest_namespace.tensorboard_log_dir:', rest_namespace.tensorboard_log_dir, ';')

    with open(logfile.name, 'rt') as f:
        prev_logfile_lines = f.readlines()

    idx = len(prev_logfile_lines) -1
    maybe_options_space = None
    while idx != 0:
        maybe_options_space = loads(prev_logfile_lines[idx])
        if isinstance(maybe_options_space, dict) and 'type' in maybe_options_space and maybe_options_space['type'] == 'options_space':
            break
        maybe_options_space = None

    received_space = ParsedLine(dataset=rest_namespace.dataset,
               epoch=0,
               value='value',
               epochs=250,
               transfer=rest_namespace.model_params[:rest_namespace.model_params.rpartition(' ')][1:-1],
               loss=rest_namespace.loss,
               optimizer=rest_namespace.optimizer,
               optimizer_params=rest_namespace.optimizer,
               base='transfer')
    if maybe_options_space is None:
        # First go!
        # Generate the entire options_space
            # this will be logged and then loaded back in
        if try_all == True:
            for m in valid_models:
                for l in SUPPORTED_LOSSES:
                    for o in SUPPORTED_OPTIMIZERS:
                        received_space.append(
                            ParsedLine(dataset=rest_namespace.dataset,
                                       epoch=0,
                                       value='value',
                                       epochs=250,
                                       transfer=m,
                                       loss=l,
                                       optimizer=o,
                                       optimizer_params=rest_namespace.optimizer,
                                       base='transfer')
                                       )

        options_space = {
            'type': 'options_space',
            'last_idx': 0,
            'space': [
                received_space
            ]
        }
    else:

        # Or if the `maybe_options_space` is not None, then it might look like:
        options_space = maybe_options_space
        options_space['space'] = map(ParsedLine, options_space['space'])
        options_space['last_idx'] = options_space['last_idx'] + 1 % len(options_space['space'])
        '''options_space = {
            'type': 'options_space',
            'last_idx': 2,
            'space': [
                ParsedLine(dataset='refuge',
                           epoch=64,
                           value='value',
                           epochs=250,
                           transfer='Resnet50',
                           loss='BinaryCrossentropy',
                           optimizer='Adam',
                           optimizer_params={'lr': 1e-3},
                           base='transfer'),
                ParsedLine(dataset='refuge',
                           epoch=64,
                           value='value',
                           epochs=250,
                           transfer='MobileNet',
                           loss='CategoricalCrossentropy',
                           optimizer='Nestrov',
                           optimizer_params={'lr': 1e-5},
                           base='transfer')
            ]
        }
        '''
        # prev_metrics = get_metrics((path.join(rest_namespace.tensorboard_log_dir, 'validation'),), prefix='epoch_')
        # prev_options = parse_line(rest_namespace.tensorboard_log_dir)

    log(options_space)
    return options_space['space']['last_idx']

def _new_modify_options(parsed_line, rest):
    if parsed_line is None:
        return
    upsert_rest_arg = partial(_upsert_cli_arg, cli=rest)
    for k in parsed_line._fields:
        upsert_rest_arg(arg=k, value=parsed_line[k])


def _execute_command(key, log, next_key, options, dry_run, rest):
    print('-------------------------------------------\n'
          '|                {cmd}ing…                |\n'
          '-------------------------------------------'.format(cmd=rest[0]), sep='')

    if rest[0] != 'train':
        raise NotImplementedError

    if dry_run:
        just = 10
        print(#'key:'.ljust(just), key, ';\n',
              'log:'.ljust(just), log, ';\n',
              #'next_key:'.ljust(just), next_key, ';\n',
              'options:'.ljust(just), dumps(options), ';\n',
              'dry_run:'.ljust(just), dry_run, ';\n',
              'rest:'.ljust(just), dumps(rest), ';', sep='')
    else:
        err, cli_resp = _handle_rest(key, locals().get('next_key'), rest, options)
        if err is not None:
            if environ.get('NO_EXCEPTIONS'):
                print(err, file=stderr)
            else:
                raise err
        print('cli_resp:', cli_resp, ';')

    print('-------------------------------------------\n'
          '|            finished {cmd}ing.           |\n'
          '-------------------------------------------'.format(cmd=rest[0]), sep='')

    if 'next_key' in locals():
        if next_key in options[key][0]:
            options[key][0][next_key] += 0.5
        else:
            options[key][0][next_key] = 0
        if '_next_key' in options:
            del options['_next_key']
        log({'options': options})


def _prepare_options(key, log, logfile, options, rest):

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

    options = get_sorted_options(options)
    next_key = next(iter(options[key][0].keys()))
    options[key][0][next_key] += 0.5
    options['_next_key'] = next_key
    log({'options': options})
    upsert_rest_arg = partial(_upsert_cli_arg, cli=rest)
    for k, v in options.items():
        if not k.startswith('_'):
            value = next(iter(v[0].keys()))

            if k == 'models':
                _handle_model_change(rest, upsert_rest_arg, value)
            else:
                upsert_rest_arg(arg=k, value=value)
    return next_key


def _handle_model_change(rest, upsert_rest_arg, model):
    namespace = ml_glaucoma.cli_options.parser.cli_handler(rest, return_namespace=True)

    gin_file = path.join(mkdtemp(prefix='gin_', dir=gettempdir()), 'applications.gin')
    copyfile(src=path.join(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))),
                           'model_configs', 'applications.gin'),
             dst=gin_file)

    upsert_rest_arg(
        arg='--model_file',
        value=gin_file
    )
    model_dir = namespace.model_dir
    _maybe_suffix = model_dir.rpartition('_')[2]
    _maybe_suffix = _maybe_suffix if _maybe_suffix.startswith('again') else None
    if _maybe_suffix:
        _maybe_suffix = '{}{:03d}'.format(_maybe_suffix[:len('again')],
                                          int(_maybe_suffix[len('again'):]) + 1)
    _join_with = lambda: '_'.join(filter(None, (namespace.dataset[0].replace('refuge', 'gon'),
                                                model,
                                                namespace.optimizer, namespace.loss,
                                                'epochs', '{:03d}'.format(namespace.epochs),
                                                _maybe_suffix)))
    model_dir = path.join(path.dirname(model_dir), _join_with())
    upsert_rest_arg(
        arg='--model_dir',
        value=model_dir
    )
    upsert_rest_arg(
        arg='--model_param',
        value='application = "{model}"'.format(model=model)
    )


def _upsert_cli_arg(arg, value, cli):
    if not arg.startswith('-'):
        arg = '--{arg}'.format(arg=arg)
    try:
        idx = cli.index(arg)
        cli[idx + 1] = value
    except ValueError:
        cli += [arg, value]
    return cli


def _handle_rest(key, next_key, rest, options):
    assert rest[0] == 'train'

    upsert_rest_arg = partial(_upsert_cli_arg, cli=rest)

    namespace = ml_glaucoma.cli_options.parser.cli_handler(rest, return_namespace=True)

    model_dir = namespace.model_dir

    if model_dir is None:
        logger.warn('model_dir is None, so not incrementing it')
    else:
        _increment_directory_suffixes(model_dir, namespace, upsert_rest_arg)

    # Replace with pipeline argument. Change next line with more complicated—i.e.: more than 1 option change—mods.
    if next_key is not None:
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


def _increment_directory_suffixes(model_dir, namespace, upsert_rest_arg):
    tensorboard_log_dir = namespace.tensorboard_log_dir
    reversed_log_dir = model_dir[::-1]
    suffix = int(''.join(takewhile(lambda s: s.isdigit(), reversed_log_dir))[::-1] or 0)
    suffix_s = '{:03d}'.format(suffix)
    if not reversed_log_dir.startswith(reversed_log_dir[:len(suffix_s)] + '_again'[::-1]):
        suffix = 0
        suffix_s = '{:03d}'.format(suffix)
    run = suffix + 1
    print('------------------------\n'
          '|        RUN {:3d}       |\n'
          '------------------------'.format(run), sep='')
    run_s = '{:03d}'.format(suffix)

    print('model_dir:'.ljust(25), model_dir)
    print('tensorboard_log_dir:'.ljust(25), tensorboard_log_dir)

    if model_dir.endswith(suffix_s):
        tensorboard_log_dir = ''.join((tensorboard_log_dir[:-len(suffix_s)], run_s))
        model_dir = ''.join((model_dir[:-len(suffix_s)], run_s))
    else:
        tensorboard_log_dir = '{}_again{}'.format(tensorboard_log_dir, suffix_s)
        model_dir = '{}_again{}'.format(model_dir, suffix_s)

    suffix, model_dir = _get_next_avail_dir(model_dir)
    _, tensorboard_log_dir = _get_next_avail_dir(tensorboard_log_dir, suffix)

    # Replace with incremented dirs
    for arg in 'model_dir', 'tensorboard_log_dir':
        upsert_rest_arg(arg=arg, value=locals()[arg])


def _get_next_avail_dir(directory, starting_suffix=None):  # type: (str, int) -> (int, str)
    suffix = starting_suffix

    if starting_suffix is not None:
        suffix_s = '{:03d}'.format(starting_suffix)
        directory, _, fname = directory.rpartition(path.sep)
        directory = path.join(directory, ''.join((fname[:-len(suffix_s)], suffix_s)))

    while path.isdir(directory) and len(listdir(directory)) > 0:
        directory, _, fname = directory.rpartition(path.sep)
        suffix = int(''.join(takewhile(lambda s: s.isdigit(), fname[::-1]))[::-1] or 0) + 1
        suffix_s = '{:03d}'.format(suffix)
        assert suffix < 999
        directory = path.join(directory, ''.join((fname[:-len(suffix_s)], suffix_s)))

    return suffix, directory
