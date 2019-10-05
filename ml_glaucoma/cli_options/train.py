from os import environ

from yaml import load as yaml_load

from ml_glaucoma import callbacks as callbacks_module
from ml_glaucoma import runners
from ml_glaucoma.cli_options.base import Configurable
from ml_glaucoma.utils.helpers import get_upper_kv

if environ['TF']:
    import tensorflow as tf

    valid_callbacks = get_upper_kv(tf.keras.callbacks)
    valid_callbacks.update(get_upper_kv(callbacks_module))
    SUPPORTED_CALLBACKS = tuple(valid_callbacks.keys())
elif environ['TORCH']:
    valid_callbacks = {}
    SUPPORTED_CALLBACKS = tuple()
else:
    valid_callbacks = {}
    SUPPORTED_CALLBACKS = tuple()


class ConfigurableTrain(Configurable):
    description = 'Train model'

    def __init__(self, problem, model_fn, optimizer, lr_schedule=None, class_weight=None, callbacks=None):
        super(ConfigurableTrain, self).__init__(
            class_weight=class_weight,
            problem=problem,
            model_fn=model_fn,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            callbacks=callbacks)

    def fill_self(self, parser):
        parser.add_argument(
            '-b', '--batch_size', default=32, type=int,
            help='size of each batch')
        parser.add_argument(
            '-e', '--epochs', default=20, type=int,
            help='number of epochs to run training from')
        parser.add_argument(
            '--class-weight', default=None, type=yaml_load,
            help='Optional dictionary mapping class indices (integers)'
                 'to a weight (float) value, used for weighting the loss function'
                 '(during training only).'
                 'This can be useful to tell the model to'
                 '"pay more attention" to samples from'
                 'an under-represented class.'
        )
        parser.add_argument(
            '--callback', nargs='*', dest='callbacks',
            choices=SUPPORTED_CALLBACKS, default=['AUC'],
            help='Keras callback function(s) to use. Extends default callback list.'
        )
        parser.add_argument(
            '--model_dir',
            help='model directory in which to save weights and '
                 'tensorboard summaries')
        parser.add_argument(
            '-c', '--checkpoint_freq', type=int, default=5,
            help='epoch frequency at which to save model weights')
        parser.add_argument(
            '--summary_freq', type=int, default=10,
            help='batch frequency at which to save tensorboard summaries')
        parser.add_argument(
            '-tb', '--tb_log_dir',
            help='tensorboard_log_dir (defaults to model_dir)')
        parser.add_argument(
            '--write_images', action='store_true',
            help='whether or not to write images to tensorboard')

    def build_self(self, problem, batch_size, epochs, model_fn, optimizer, model_dir,
                   callbacks, checkpoint_freq, summary_freq, lr_schedule, tb_log_dir,
                   class_weight, write_images, **_kwargs):
        return runners.train(
            callbacks=[] if callbacks is None else list(map(lambda callback: valid_callbacks[callback], callbacks)),
            problem=problem,
            batch_size=batch_size,
            epochs=epochs,
            model_fn=model_fn,
            optimizer=optimizer,
            class_weight=class_weight,
            model_dir=model_dir,
            checkpoint_freq=checkpoint_freq,
            summary_freq=summary_freq,
            lr_schedule=lr_schedule,
            tensorboard_log_dir=tb_log_dir,
            write_images=write_images,
        )
