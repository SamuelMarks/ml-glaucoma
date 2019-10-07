# See ml-glaucoma/ml_glaucoma/cli_options/train.py for other hyperparameters
from __future__ import absolute_import

from os import environ

from yaml import load as yaml_load

from ml_glaucoma import get_logger
from ml_glaucoma.cli_options import ConfigurableMapFn

if environ['TF']:
    from ml_glaucoma.cli_options.hyperparameters.tf_keras import *
elif environ['TORCH']:
    from ml_glaucoma.cli_options.hyperparameters.torch import *
else:
    from ml_glaucoma.cli_options.hyperparameters.other import *

logger = get_logger(__file__.partition('.')[0])


class ConfigurableProblem(ConfigurableProblemBase):
    def __init__(self, builders, map_fn=None):
        if map_fn is None:
            map_fn = ConfigurableMapFn()
        super(ConfigurableProblem, self).__init__(
            builders=builders, map_fn=map_fn)

    def fill_self(self, parser):
        parser.add_argument(
            '-l', '--loss', choices=SUPPORTED_LOSSES,
            default='BinaryCrossentropy',
            help='loss function to use')
        parser.add_argument(
            '-m', '--metrics', nargs='*',
            choices=SUPPORTED_METRICS, default=['AUC'],
            help='metric functions to use')
        parser.add_argument(
            '-pt', '--precision_thresholds',
            default=[0.5], type=float, nargs='*',
            help='precision thresholds', )
        parser.add_argument(
            '-rt', '--recall_thresholds', default=[0.5], type=float, nargs='*',
            help='recall thresholds')
        parser.add_argument(
            '--shuffle_buffer', default=1024, type=int,
            help='buffer used in tf.data.Dataset.shuffle')
        parser.add_argument(
            '--use_inverse_freq_weights', action='store_true',
            help='weight loss according to inverse class frequency')


class ConfigurableModelFn(Configurable):
    def __init__(self):
        super(ConfigurableModelFn, self).__init__()

    def fill_self(self, parser):
        parser.add_argument(
            '--model_file', nargs='*',
            help='gin files for model definition. Should define `model_fn` '
                 'macro either here or in --gin_param')
        parser.add_argument(
            '--model_param', nargs='*',
            help='gin_params for model definition. Should define `model_fn` '
                 'macro either here or in --gin_file')

    def build_self(self, model_file, model_param, **kwargs):
        import gin

        gin.parse_config_files_and_bindings(model_file, model_param)
        return gin.query_parameter('%model_fn').configurable.fn_or_cls


class ConfigurableOptimizer(Configurable):
    def __init__(self):
        super(ConfigurableOptimizer, self).__init__()

    def fill_self(self, parser):
        parser.add_argument(
            '-o', '--optimizer', default='Adam', choices=SUPPORTED_OPTIMIZERS,
            help='class name of optimizer to use')
        parser.add_argument(
            '-lr', '--learning_rate', default=1e-3, type=float,
            help='base optimizer learning rate')
        parser.add_argument(
            '--optimiser_args', default={}, type=yaml_load,
            help='Extra optimiser args, e.g.: \'{epsilon: 1e-7, amsgrad: true}\''
        )

    def build(self, optimizer, learning_rate, optimiser_args, **kwargs):
        if 'lr' in optimiser_args:
            logger.warn('Learning rate is being replaced by `--learning_rate` value or its default')

        optimiser_args['lr'] = learning_rate
        return valid_optimizers[optimizer](**optimiser_args)

    def build_self(self, learning_rate, optimiser_args, exp_lr_decay, **kwargs):
        raise NotImplementedError


class ConfigurableExponentialDecayLrSchedule(Configurable):
    def __init__(self):
        super(ConfigurableExponentialDecayLrSchedule, self).__init__()

    def fill_self(self, parser):
        parser.add_argument(
            '--exp_lr_decay', default=None,
            help='exponential learning rate decay factor applied per epoch, '
                 'e.g. 0.98. None is interpreted as no decay')

    def build_self(self, learning_rate, exp_lr_decay, **kwargs):
        if exp_lr_decay is None:
            return None
        else:
            from ml_glaucoma.callbacks.exponential_decay_lr_schedule import ExponentialDecayLrSchedule
            return ExponentialDecayLrSchedule(learning_rate, exp_lr_decay)
