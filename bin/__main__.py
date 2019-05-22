from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools
import numpy as np
import tensorflow as tf
from ml_glaucoma import problems as p
from ml_glaucoma import runners

SUPPORTED_LOSSES = (
    'BinaryCrossentropy',
)

SUPPORTED_METRICS = (
    'F1',
    'AUC',
    'BinaryAccuracy',
)

# use --recall_thresholds and --precision_thresholds for Precision/Recall

SUPPORTED_OPTIMIZERS = tuple(
    k for k in dir(tf.keras.optimizers) if
    not k.startswith('_') and k not in ('get', 'deserialize'))


class Configurable(object):
    def __init__(self, **children):
        self._children = children
        for v in children.values():
            assert(v is None or isinstance(v, Configurable))

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



class ConfigurableBuilder(Configurable):
    def __init__(self):
        super(ConfigurableBuilder, self).__init__()

    def fill_self(self, parser):
        parser.add_argument(
            '-ds', '--dataset', help='dataset key', choices=('bmes', 'refuge'), default='refuge')
        parser.add_argument(
            '--data_dir', help='root directory to store processed tfds records')
        parser.add_argument(
            '--download_dir', help='directory to store downloaded files')
        parser.add_argument(
            '--extract_dir', help='directory where extracted files are stored')
        parser.add_argument(
            '--manual_dir', help='directory where manually downloaded files are saved')
        parser.add_argument(
            '--download_mode', help='tfds.GenerateMode', choices=('reuse_dataset_if_exists', 'reuse_cache_if_exists', 'force_redownload'))
        parser.add_argument(
            '-r', '--resolution', help='image resolution', nargs=2, default=(256, 256), type=int)
        parser.add_argument(
            '--gray_on_disk', help='whether or not to save data as grayscale on disk', action='store_true')

    def build_self(
            self, dataset, data_dir, download_dir, extract_dir, manual_dir,
            download_mode, resolution, gray_on_disk, **kwargs):
        import tensorflow_datasets as tfds
        config_kwargs = dict(rgb=not gray_on_disk, resolution=resolution)
        if dataset == 'bmes':
            from ml_glaucoma.problems import bmes
            builder = bmes.Bmes(
                bmes.BmesConfig(**config_kwargs),
                data_dir=data_dir)
        elif dataset == 'refuge':
            from ml_glaucoma.problems import refuge
            builder = refuge.Refuge(
                refuge.RefugeConfig(**config_kwargs),
                data_dir=data_dir)
        else:
            raise NotImplementedError

        p.download_and_prepare(
            builder=builder,
            download_config=tfds.download.DownloadConfig(
                extract_dir=extract_dir, manual_dir=manual_dir,
                download_mode=download_mode),
            download_dir=download_dir)
        return builder


class ConfigurableMapFn(Configurable):
    def fill_self(self, parser):
        parser.add_argument(
            '-fv', '--maybe_vertical_flip', help='randomly flip training input vertically', action='store_true')
        parser.add_argument(
            '-fh', '--maybe_horizontal_flip', help='randomly flip training input horizontally', action='store_true')
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
    def __init__(self, builder, map_fn=None):
        if map_fn is None:
            map_fn = ConfigurableMapFn()
        super(ConfigurableProblem, self).__init__(
            builder=builder, map_fn=map_fn)

    def fill_self(self, parser):
        parser.add_argument(
            '-l', '--loss', help='loss function to use', choices=SUPPORTED_LOSSES, default='BinaryCrossentropy')
        parser.add_argument(
            '-m', '--metrics', nargs='*', help='metric functions to use', choices=SUPPORTED_METRICS, default=['AUC'])
        parser.add_argument(
            '-pt', '--precision_thresholds', help='precision thresholds', default=[0.5], type=float, nargs='*')
        parser.add_argument(
            '-rt', '--recall_thresholds', help='recall thresholds', default=[0.5], type=float, nargs='*')
        parser.add_argument(
            '--shuffle_buffer', help='buffer used in tf.data.Dataset.shuffle', default=1024, type=int)
        parser.add_argument(
            '--use_inverse_freq_weights', help='weight loss according to inverse class frequency', action='store_true')

    def build_self(
            self, builder, map_fn, loss, metrics,
            precision_thresholds, recall_thresholds,
            shuffle_buffer, use_inverse_freq_weights,
            **kwargs):
        # from ml_glaucoma.metrics import BinaryPrecision
        # from ml_glaucoma.metrics import BinaryRecall
        # from ml_glaucoma.metrics import BinaryAdapter

        metrics = [
            tf.keras.metrics.deserialize(dict(class_name=m, config={}))
            for m in metrics]
        # multiple threshold values don't seem to work for metrics
        metrics.extend(
            [tf.keras.metrics.Precision(
                thresholds=[t],
                name='precision%d' % int((100*t)))
             for t in precision_thresholds])
        metrics.extend(
            [tf.keras.metrics.Recall(
                thresholds=[r],
                name='recall%d' % (int(100*r)))
             for r in recall_thresholds])

        return p.TfdsProblem(
            builder=builder,
            loss=tf.keras.losses.deserialize(dict(class_name=loss, config={})),
            metrics=metrics,
            map_fn=map_fn,
            shuffle_buffer=shuffle_buffer,
            use_inverse_freq_weights=use_inverse_freq_weights)


class ConfigurableModelFn(Configurable):
    def __init__(self):
        super(ConfigurableModelFn, self).__init__()

    def fill_self(self, parser):
        import gin
        parser.add_argument('--model_file', nargs='*', help='gin files for model definition. Should define `model_fn` macro either here or in --gin_param')
        parser.add_argument('--model_param', nargs='*', help='gin_params for model definition. Should define `model_fn` macro either here or in --gin_file')


    def build_self(self, model_file, model_param, **kwargs):
        import gin
        gin.parse_config_files_and_bindings(model_file, model_param)
        return gin.query_parameter('%model_fn').configurable.fn_or_cls


class ConfigurableOptimizer(Configurable):
    def __init__(self):
        super(ConfigurableOptimizer, self).__init__()

    def fill_self(self, parser):
        parser.add_argument('-o', '--optimizer', default='Adam', choices=SUPPORTED_OPTIMIZERS, help='class name of optimizer to use')
        parser.add_argument('-lr', '--learning_rate', default=1e-3, help='base optimizer learning rate', type=float)

    def build(self, optimizer, learning_rate, **kwargs):
        return getattr(tf.keras.optimizers, optimizer)(lr=learning_rate)


class ConfigurableExponentialDecayLrSchedule(Configurable):
    def __init__(self):
        super(ConfigurableExponentialDecayLrSchedule, self).__init__()

    def fill_self(self, parser):
        parser.add_argument('--exp_lr_decay', help='exponential learning rate decay factor applied per epoch, e.g. 0.98. None is interpreted as no decay', default=None)

    def build_self(self, learning_rate, exp_lr_decay, **kwargs):
        if exp_lr_decay is None:
            return None
        else:
            from ml_glaucoma.callbacks import exponential_decay_lr_schedule
            return exponential_decay_lr_schedule(learning_rate, exp_lr_decay)


class ConfigurableTrain(Configurable):
    def __init__(self, problem, model_fn, optimizer, lr_schedule=None):
        super(ConfigurableTrain, self).__init__(
            problem=problem, model_fn=model_fn, optimizer=optimizer, lr_schedule=lr_schedule)

    def fill_self(self, parser):
        parser.add_argument(
            '-b', '--batch_size', default=32, type=int, help='size of each batch')
        parser.add_argument(
            '-e', '--epochs', default=20, type=int, help='number of epochs to run training from')
        parser.add_argument(
            '--model_dir', help='model directory in which to save weights and tensorboard summaries')
        parser.add_argument(
            '-c', '--checkpoint_freq', type=int, help='epoch frequency at which to save model weights', default=5)
        parser.add_argument(
            '--summary_freq', type=int, help='batch frequency at which to save tensorboard summaries', default=10)
        parser.add_argument(
            '-tb', '--tb_log_dir', help='tensorboard_log_dir (defaults to model_dir)')
        parser.add_argument(
            '--write_images', action='store_true', help='whether or not to write images to tensorboard')

    def build_self(self, problem, batch_size, epochs, model_fn, optimizer, model_dir, checkpoint_freq, summary_freq, lr_schedule, tb_log_dir, write_images, **kwargs):
        return runners.train(
            problem=problem,
            batch_size=batch_size,
            epochs=epochs,
            model_fn=model_fn,
            optimizer=optimizer,
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
            '-b', '--batch_size', default=32, type=int, help='size of each batch')
        parser.add_argument(
            '--model_dir', help='model directory in which to save weights and tensorboard summaries')

    def build_self(self, porblem, batch_size, model_fn, optimizer, model_dir, **kwargs):
        from ml_glaucoma import runners
        return runners.evaluate(
            problem=problem,
            batch_size=batch_size,
            model_fn=model_fn,
            optimizer=optimizer,
            model_dir=model_dir,
        )


def main():
    from argparse import ArgumentParser
    commands = {}
    parser = ArgumentParser(
        description='CLI for a Glaucoma diagnosing CNN and preparing data for such')
    subparsers = parser.add_subparsers(dest='command')

    builder = ConfigurableBuilder()
    map_fn = ConfigurableMapFn()
    problem = ConfigurableProblem(builder, map_fn)
    model_fn = ConfigurableModelFn()
    optimizer = ConfigurableOptimizer()
    lr_schedule = ConfigurableExponentialDecayLrSchedule()

    train = ConfigurableTrain(problem, model_fn, optimizer, lr_schedule)
    evaluate = ConfigurableEvaluate(problem, model_fn, optimizer)

    # DOWNLOAD
    download_parser = subparsers.add_parser(
        'download', help='Download and prepare required data')
    builder.fill(download_parser)
    commands['download'] = builder

    vis_parser = subparsers.add_parser(
        'vis', help='Download and prepare required data')
    problem.fill(vis_parser)
    commands['vis'] = problem.map(runners.vis)

    # TRAIN
    train_parser = subparsers.add_parser('train', help='Train model')
    train.fill(train_parser)
    commands['train'] = train

    # EVALUATE
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    train.fill(evaluate_parser)
    commands['evaluate'] = evaluate

    kwargs = dict(parser.parse_args()._get_kwargs())
    command = kwargs.pop('command')
    if command is None:
        raise ReferenceError(
            'You must specify a command. Append `--help` for details.')

    return commands[command].build(**kwargs)


if __name__ == '__main__':
    main()
