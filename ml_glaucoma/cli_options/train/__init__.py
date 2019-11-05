from os import environ

from yaml import load as yaml_load

import ml_glaucoma.runners
from ml_glaucoma.cli_options.base import Configurable

if environ['TF']:
    from ml_glaucoma.cli_options.train.tf_keras import *
elif environ['TORCH']:
    from ml_glaucoma.cli_options.train.torch import *
else:
    from ml_glaucoma.cli_options.train.other import *


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
        parser.add_argument(
            '--seed', type=int, help='Set the seed, combine with `--disable-gpu` to disable GPU for added determinism'
        )
        parser.add_argument(
            '--disable-gpu', action='store_true',
            help='Set the seed, combine with `--disable-gpu` to disable GPU for added determinism'
        )
        parser.add_argument(
            '--continuous', action='store_true',
            help='after each successful train, run again'
        )
        parser.add_argument(
            '--delete-lt', type=float,
            help='delete *.h5 files that are less than this threshold'
        )

    def build_self(self, problem, batch_size, epochs, model_fn, optimizer, model_dir,
                   callbacks, checkpoint_freq, summary_freq, lr_schedule, tb_log_dir,
                   class_weight, write_images, seed, disable_gpu, continuous, **_kwargs):
        if disable_gpu:
            environ['CUDA_VISIBLE_DEVICES'] = '-1'

        if seed:
            environ['PYTHONHASHSEED'] = str(seed)
            import random
            random.seed(seed)

            # 3. Set `numpy` pseudo-random generator at a fixed value
            import numpy as np
            np.random.seed(seed)

            # 4. Set `tensorflow` pseudo-random generator at a fixed value
            if environ['TF']:
                import tensorflow as tf
                import tensorflow.keras.backend as K
                tf.compat.v1.set_random_seed(seed)
                session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
                sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
                K.set_session(sess)
            elif environ['TORCH']:
                import torch

                torch.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        return ml_glaucoma.runners.train(
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
            continuous=continuous
        )
