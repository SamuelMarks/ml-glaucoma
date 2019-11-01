from sys import modules

import ml_glaucoma.runners
from ml_glaucoma import __version__
from ml_glaucoma.cli_options.augment import ConfigurableMapFn
from ml_glaucoma.cli_options.evaluate import ConfigurableEvaluate
from ml_glaucoma.cli_options.hyperparameters import (ConfigurableProblem, ConfigurableModelFn,
                                                     ConfigurableOptimizer, ConfigurableExponentialDecayLrSchedule)
from ml_glaucoma.cli_options.logparser import ConfigurableLogParser, log_parser
from ml_glaucoma.cli_options.prepare import ConfigurableBuilders
from ml_glaucoma.cli_options.train import ConfigurableTrain


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

    return _parser, _commands
