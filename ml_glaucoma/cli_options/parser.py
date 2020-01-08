from sys import modules

import ml_glaucoma.runners
from ml_glaucoma import __version__
from ml_glaucoma.cli_options.augment import ConfigurableMapFn
from ml_glaucoma.cli_options.evaluate import ConfigurableEvaluate
from ml_glaucoma.cli_options.hyperparameters import (ConfigurableProblem, ConfigurableModelFn,
                                                     ConfigurableOptimizer, ConfigurableExponentialDecayLrSchedule)
from ml_glaucoma.cli_options.info import ConfigurableInfo
from ml_glaucoma.cli_options.logparser import ConfigurableLogParser
from ml_glaucoma.cli_options.pipeline import ConfigurablePipeline
from ml_glaucoma.cli_options.prepare import ConfigurableBuilders
from ml_glaucoma.cli_options.train import ConfigurableTrain


def _reparse_cli(cmd):  # type: ([str] or None) -> [str] or None
    """
    Namespace parsing will cause all nargs to be put into ['--carg', ['a','b']] lists,
     to pass it back this needs to revert to ['--carg', 'a', '--carg', 'b')
    """
    if cmd is None:
        return cmd

    skip = False
    len_cmd = len(cmd)
    a = []
    for i, c in enumerate(cmd):
        j = i + 1
        if j < len_cmd and isinstance(cmd[j], list):
            for cj in cmd[j]:
                a += [c, str(cj)]
            skip = True
        elif skip:
            skip = False
        else:
            a.append(str(c))

    return a


def cli_handler(cmd=None, return_namespace=False):
    parser, commands = get_parser()

    args, rest = parser.parse_known_args(cmd if cmd is None
                                         else tuple(map(lambda c: c if c is None else str(c),
                                                        _reparse_cli(cmd))))

    if return_namespace:
        cmd = commands[args.command]

        kwargs = vars(args)
        cmd.set_defaults(kwargs)
        for attr in dir(args):
            if not attr.startswith('_'):
                setattr(args, attr, kwargs[attr])

        return args

    kwargs = dict(vars(args), rest=rest)

    command = kwargs.pop('command')
    if command is None:
        raise ReferenceError('You must specify a command. Append `--help` for details.')

    cmd = commands[command]
    cmd.set_defaults(kwargs)

    return cmd.build(**kwargs)


def get_parser():
    from argparse import ArgumentParser

    _commands = {}
    _parser = ArgumentParser(
        prog=None if globals().get('__spec__') is None else 'python -m {}'.format(__spec__.name.partition('.')[0]),
        description='CLI for a Glaucoma diagnosing CNN'
    )
    _parser.add_argument('--version', action='version',
                         version='{} {}'.format(modules[__name__].__package__.partition('.')[0], __version__))
    subparsers = _parser.add_subparsers(dest='command')

    builders_config = ConfigurableBuilders()
    map_fn_config = ConfigurableMapFn()
    problem_config = ConfigurableProblem(builders_config, map_fn_config)
    model_fn_config = ConfigurableModelFn()
    optimizer_config = ConfigurableOptimizer()
    lr_schedule = ConfigurableExponentialDecayLrSchedule()

    train_config = ConfigurableTrain(problem_config, model_fn_config, optimizer_config, lr_schedule)
    evaluate_config = ConfigurableEvaluate(problem_config, model_fn_config, optimizer_config)

    # DOWNLOAD
    download_parser = subparsers.add_parser(
        'download', help='Download and prepare required data')
    builders_config.fill(download_parser)
    _commands['download'] = builders_config

    # VISUALISE
    vis_parser = subparsers.add_parser('vis', help='Visualise data')
    problem_config.fill(vis_parser)
    _commands['vis'] = problem_config.map(ml_glaucoma.runners.vis)

    # TRAIN
    train_parser = subparsers.add_parser('train', help=ConfigurableTrain.description)
    train_config.fill(train_parser)
    _commands['train'] = train_config

    # EVALUATE
    evaluate_parser = subparsers.add_parser('evaluate', help=ConfigurableEvaluate.description)
    train_config.fill(evaluate_parser)
    _commands['evaluate'] = evaluate_config

    # PARSE
    parser_parser = subparsers.add_parser('parser', help=ConfigurableLogParser.description)
    log_parser_configurable = ConfigurableLogParser()
    log_parser_configurable.fill(parser_parser)
    _commands['parser'] = log_parser_configurable

    # INFO
    info_parser = subparsers.add_parser('info', help=ConfigurableInfo.description)
    info_configurable = ConfigurableInfo()
    info_configurable.fill(info_parser)
    _commands['info'] = info_configurable

    # PIPELINE
    pipeline_parser = subparsers.add_parser('pipeline', help=ConfigurablePipeline.description)
    pipeline_configurable = ConfigurablePipeline()
    pipeline_configurable.fill(pipeline_parser)
    _commands['pipeline'] = pipeline_configurable

    return _parser, _commands
