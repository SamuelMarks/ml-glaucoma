from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools

import tensorflow as tf
import yaml

from ml_glaucoma import losses as losses_module, callbacks as callbacks_module
from ml_glaucoma import problems as p
from ml_glaucoma import runners
from ml_glaucoma.utils.helpers import get_upper_kv

valid_losses = {loss: getattr(tf.keras.losses, loss)
                for loss in dir(tf.keras.losses)
                if not loss.startswith('_')}
valid_losses.update({loss_name: getattr(tf.losses, loss_name)
                     for loss_name in dir(tf.losses)
                     if not loss_name.startswith('_') and loss_name == 'Reduction'})
valid_losses.update(get_upper_kv(losses_module))
SUPPORTED_LOSSES = tuple(valid_losses.keys())

valid_metrics = {metric: getattr(tf.keras.metrics, metric)
                 for metric in dir(tf.keras.metrics)
                 if not metric.startswith('_') and metric not in frozenset(('serialize', 'deserialize', 'get'))}
SUPPORTED_METRICS = tuple(valid_metrics.keys())

# use --recall_thresholds and --precision_thresholds for Precision/Recall

valid_optimizers = get_upper_kv(tf.keras.optimizers)
SUPPORTED_OPTIMIZERS = tuple(valid_optimizers.keys())

valid_callbacks = get_upper_kv(tf.keras.callbacks)
valid_callbacks.update(get_upper_kv(callbacks_module))
SUPPORTED_CALLBACKS = tuple(valid_callbacks.keys())


class Configurable(object):
    def __init__(self, **children):
        self._children = children
        for v in children.values():
            assert (v is None or isinstance(v, Configurable))

    def fill(self, parser):
        for child in self._children.values():
            if child is not None:
                child.fill(parser)
        self.fill_self(parser)

    def build(self, **kwargs):
        children_values = {
            k: None if child is None else child.build(**kwargs)
            for k, child in self._children.items()}
        kwargs.update(children_values)
        return self.build_self(**kwargs)

    @abc.abstractmethod
    def fill_self(self, parser):
        raise NotImplementedError

    @abc.abstractmethod
    def build_self(self, **kwargs):
        raise NotImplementedError

    def map(self, fn):
        return MappedConfigurable(self, fn)


class MappedConfigurable(Configurable):
    def __init__(self, base, fn):
        super(MappedConfigurable, self).__init__(base=base)
        self.fn = fn

    def fill_self(self, **kwargs):
        pass

    def build_self(self, base, **kwargs):
        return self.fn(base)


class ConfigurableBuilders(Configurable):
    def __init__(self):
        super(ConfigurableBuilders, self).__init__()

    def fill_self(self, parser):
        parser.add_argument(
            '-ds', '--dataset', choices=('bmes', 'refuge'), default=['refuge'],
            nargs='+',
            help='dataset key', )
        parser.add_argument(
            '--data_dir',
            help='root directory to store processed tfds records')
        parser.add_argument(
            '--download_dir', help='directory to store downloaded files')
        parser.add_argument(
            '--extract_dir', help='directory where extracted files are stored')
        parser.add_argument(
            '--manual_dir',
            help='directory where manually downloaded files are saved')
        parser.add_argument(
            '--download_mode',
            choices=(
                'reuse_dataset_if_exists',
                'reuse_cache_if_exists',
                'force_redownload'),
            help='tfds.GenerateMode')
        parser.add_argument(
            '-r', '--resolution', nargs=2, default=(256, 256), type=int,
            help='image resolution')
        parser.add_argument(
            '--gray_on_disk', action='store_true',
            help='whether or not to save data as grayscale on disk')
        parser.add_argument(
            '--bmes_init', action='store_true', help='initial bmes get_data')
        parser.add_argument(
            '--bmes_parent_dir', help='parent directory of bmes data')

    def build_self(self, dataset, data_dir, download_dir, extract_dir, manual_dir,
                   download_mode, resolution, gray_on_disk, bmes_init, bmes_parent_dir, **kwargs):
        import tensorflow_datasets as tfds
        builders = []
        for ds in set(dataset):  # remove duplicates
            if ds == 'bmes':
                from ml_glaucoma.tfds_builders import bmes

                builder_factory = bmes.get_bmes_builder
                if bmes_init:
                    from ml_glaucoma.utils.get_data import get_data

                    if manual_dir is None:
                        raise ValueError(
                            '`manual_dir` must be provided if doing bmes_init')

                    if bmes_parent_dir is None:
                        raise ValueError(
                            '`bmes_parent_dir` must be provided if doing '
                            'bmes_init')

                    get_data(bmes_parent_dir, manual_dir)
            elif ds == 'refuge':
                from ml_glaucoma.tfds_builders import refuge

                builder_factory = refuge.get_refuge_builder
            else:
                raise NotImplementedError

            builder = builder_factory(
                resolution=resolution,
                rgb=not gray_on_disk,
                data_dir=data_dir)

            p.download_and_prepare(
                builder=builder,
                download_config=tfds.download.DownloadConfig(
                    extract_dir=extract_dir, manual_dir=manual_dir,
                    download_mode=download_mode),
                download_dir=download_dir)
            builders.append(builder)

        return builders


class ConfigurableMapFn(Configurable):
    def fill_self(self, parser):
        parser.add_argument(
            '-fv', '--maybe_vertical_flip', action='store_true',
            help='randomly flip training input vertically')
        parser.add_argument(
            '-fh', '--maybe_horizontal_flip', action='store_true',
            help='randomly flip training input horizontally')
        parser.add_argument(
            '--gray', help='use grayscale', action='store_true')

    def build_self(self, gray, maybe_horizontal_flip, maybe_vertical_flip, **kwargs):
        val_map_fn = functools.partial(
            p.preprocess_example,
            use_rgb=not gray,
            per_image_standardization=True)

        train_map_fn = functools.partial(
            val_map_fn,
            maybe_horizontal_flip=maybe_horizontal_flip,
            maybe_vertical_flip=maybe_vertical_flip)
        map_fn = {
            'train': train_map_fn,
            'validation': val_map_fn,
            'test': val_map_fn
        }
        return map_fn


class ConfigurableProblem(Configurable):
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

    def build_self(self, builders, map_fn, loss, metrics,
                   precision_thresholds, recall_thresholds,
                   shuffle_buffer, use_inverse_freq_weights,
                   **kwargs):
        metrics = [
            (tf.keras.metrics.deserialize(dict(class_name=metric, config={})) if metric in dir(tf.keras.metrics)
             else valid_metrics[metric])
            for metric in metrics
        ]

        # multiple threshold values don't seem to work for metrics
        metrics.extend(
            [tf.keras.metrics.TruePositives(
                thresholds=[t],
                name='tp{:d}'.format(int(100 * t)))
                for t in precision_thresholds])
        metrics.extend(
            [tf.keras.metrics.FalsePositives(
                thresholds=[t],
                name='fp{:d}'.format(int(100 * t)))
                for t in precision_thresholds])
        metrics.extend(
            [tf.keras.metrics.TrueNegatives(
                thresholds=[t],
                name='tn{:d}'.format(int(100 * t)))
                for t in precision_thresholds])
        metrics.extend(
            [tf.keras.metrics.FalseNegatives(
                thresholds=[t],
                name='fn{:d}'.format(int(100 * t)))
                for t in precision_thresholds])
        metrics.append(
            tf.contrib.metrics.f1_score(
                name='f1{:d}')
        )
        metrics.extend(
            [tf.keras.metrics.Precision(
                thresholds=[t],
                name='precision{:d}'.format(int(100 * t)))
                for t in precision_thresholds])
        metrics.extend(
            [tf.keras.metrics.Recall(
                thresholds=[r],
                name='recall{:d}'.format(int(100 * r)))
                for r in recall_thresholds])

        kwargs = dict(
            loss=(tf.keras.losses.deserialize(dict(class_name=loss, config={})) if loss in dir(tf.keras.losses)
                  else valid_losses[loss]),
            metrics=metrics,
            map_fn=map_fn,
            shuffle_buffer=shuffle_buffer,
            use_inverse_freq_weights=use_inverse_freq_weights)
        if len(builders) == 1:
            return p.TfdsProblem(builder=builders[0], **kwargs)
        else:
            return p.TfdsMultiProblem(builders=builders, **kwargs)


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

    def build(self, optimizer, learning_rate, **kwargs):
        return valid_optimizers[optimizer](lr=learning_rate)

    def build_self(self, learning_rate, exp_lr_decay, **kwargs):
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


class ConfigurableTrain(Configurable):
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
            '--class-weight', default=None, type=yaml.load,
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


class ConfigurableEvaluate(Configurable):
    def __init__(self, problem, model_fn, optimizer):
        super(ConfigurableEvaluate, self).__init__(
            problem=problem, model_fn=model_fn, optimizer=optimizer)

    def fill_self(self, parser):
        parser.add_argument(
            '-b', '--batch_size', default=32, type=int,
            help='size of each batch')
        parser.add_argument(
            '--model_dir',
            help='model directory in which to save weights and tensorboard '
                 'summaries')

    def build_self(self, problem, batch_size, model_fn, optimizer, model_dir,
                   **kwargs):
        from ml_glaucoma import runners
        return runners.evaluate(
            problem=problem,
            batch_size=batch_size,
            model_fn=model_fn,
            optimizer=optimizer,
            model_dir=model_dir,
        )


def get_parser():
    from argparse import ArgumentParser

    _commands = {}
    _parser = ArgumentParser(description='CLI for a Glaucoma diagnosing CNN')
    subparsers = _parser.add_subparsers(dest='v2_command')

    builders = ConfigurableBuilders()
    map_fn = ConfigurableMapFn()
    problem = ConfigurableProblem(builders, map_fn)
    model_fn = ConfigurableModelFn()
    optimizer = ConfigurableOptimizer()
    lr_schedule = ConfigurableExponentialDecayLrSchedule()

    train = ConfigurableTrain(problem, model_fn, optimizer, lr_schedule)
    evaluate = ConfigurableEvaluate(problem, model_fn, optimizer)

    # DOWNLOAD
    download_parser = subparsers.add_parser(
        'download', help='Download and prepare required data')
    builders.fill(download_parser)
    _commands['download'] = builders

    vis_parser = subparsers.add_parser(
        'vis', help='Download and prepare required data')
    problem.fill(vis_parser)
    _commands['vis'] = problem.map(runners.vis)

    # TRAIN
    train_parser = subparsers.add_parser('train', help='Train model')
    train.fill(train_parser)
    _commands['train'] = train

    # EVALUATE
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    train.fill(evaluate_parser)
    _commands['evaluate'] = evaluate

    return _parser, _commands


def cli_v2_processor(main_parser, commands):
    kwargs = dict(main_parser.parse_args()._get_kwargs())
    command = kwargs.pop('v2_command')
    if command is None:
        raise ReferenceError(
            'You must specify a command. Append `--help` for details.')

    commands[command].build(**kwargs)
