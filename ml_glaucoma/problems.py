"""
Problems are mostly adapters operating between base datasets and keras models.

While `tensorflow_datasets.DatasetBuilder`s (see `ml_glaucoma.tfds_builders`
for implementations) provides data download, and serialization and meta-data
collection, problems provide a customizable interface for the models to be
trained. They also include metrics, losses and data augmentation preprocessing.

The `Problem` class provides the general interface, while `TfdsProblem` is a
basic implementation that leverages `tensorflow_datasets.DatasetBuilder`s.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from os import environ

from absl import logging


def download_and_prepare(builder, download_config=None, download_dir=None):
    if download_config is None:
        download_config = DownloadConfig()
    builder.download_and_prepare(
        download_dir=download_dir, download_config=download_config)


def _examples_per_epoch(builder, split):
    return int(builder.info.splits[split].num_examples)


class Problem(object):
    """
    Abstract base class representing  data and evaluation.

    See `BaseProblem` for base implementation.
    """

    @abc.abstractmethod
    def get_dataset(self, split, repeat=False, batch_size=None):
        raise NotImplementedError

    @abc.abstractmethod
    def examples_per_epoch(self, split=None):  # `split=tfds.Split.TRAIN`
        raise NotImplementedError

    @abc.abstractmethod
    def loss(self):
        raise NotImplementedError

    # -------------------------------------------------
    # Base implementations: these make possibly wrong assumptions
    @property
    def metrics(self):
        return None

    def output_spec(self):
        """Assumed to be the same as target_spec, but not necessarily."""
        return self.target_spec()

    # -------------------------------------------------
    # Base implementations: these are possibly inefficient
    def dataset_spec(self):
        """Structure of `tf.keras.layers.InputSpec` corresponding to output."""
        return dataset_spec_to_input_spec(self.get_dataset('train'))

    def input_spec(self):
        """First output (index 0) of `dataset_spec`."""
        return self.dataset_spec()[0]

    def target_spec(self):
        """Second output (index 1) of `dataset_spec`."""
        return self.dataset_spec()[1]


if environ['TF']:
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorflow.keras.layers import InputSpec
    from tensorflow_datasets.core.download import DownloadConfig


    def dataset_spec_to_input_spec(dataset, has_batch_dim=False):
        """Convert dataset output_shapes and output_types to `InputSpec`s."""
        if has_batch_dim:
            def f(shape, dtype):
                return InputSpec(shape=shape[1:], dtype=dtype)
        else:
            def f(shape, dtype):
                return InputSpec(shape=shape, dtype=dtype)
        return tf.nest.map_structure(
            f, dataset.output_shapes, dataset.output_types)


    class BaseProblem(Problem):
        """
        Almost-complete basic implementation of `Problem`.

        Requires the following methods to be implemented:
            * _get_base_dataset
            * examples_per_epoch
        """

        def __init__(self, loss, metrics, shuffle_buffer=1024,
                     map_fn=None, use_inverse_freq_weights=False,
                     class_counts=None, output_spec=None,
                     dataset_spec=None):
            """
            Base implementation of `Problem`.

            Args:
                loss: `tf.keras.losses.Loss` instance
                metrics: iterable of `tf.keras.metrics.Metric` instances
                shuffle_buffer: used in `tf.data.Dataset.shuffle`
                map_fn: optional map function/dict of split -> map_fn that is
                    applied to each example independently.
                    Used in `tf.data.Dataset.map` prior to batching.
                use_inverse_freq_weights: for classification problems, this creates
                    weights based on inverse class frequencies.
                class_counts (optional): iterable of ints, class counts. If given
                    and `use_inverse_freq_weights`, the these are the frequencies
                    used. There is no requirement that they correspond to the
                    actual class counts, but the weights will be calculated as if
                    they are. If not provided, the dataset is reduced to count the
                    relevant weights.
                output_spec: (optional) structure of `tf.keras.layers.InputSpec`s
                    corresponding to the desired output. If `None`, this is assumed
                    to be the same as `target_spec`, i.e. the spec of the second
                    output of `self.dataset_spec()`, though this may be incorrect
                    (e.g. in sparse classification problems, where outputs (logits)
                    should have 1 extra dimension compared to the sparse labels and
                    be floats).
                dataset_spec: (optional) structure of `tf.keras.layers.InputSpec`s
                    corresponding to the output of `self.get_dataset()`. Providing
                    this explicitly removes the necessity to build the dataset in
                    order to calculate it.
            """
            self._map_fn = map_fn
            self._use_inverse_freq_weights = use_inverse_freq_weights
            self._class_counts = class_counts
            if shuffle_buffer is None:
                shuffle_buffer = self.examples_per_epoch('train')
            self._shuffle_buffer = shuffle_buffer
            self._loss = loss
            self._metrics = tuple(metrics)
            self._output_spec = output_spec
            self._dataset_spec = dataset_spec

        def output_spec(self):
            """Assumed to be the same as target_spec, but not necessarily."""
            if self._output_spec is None:
                self._output_spec = self.target_spec()
            return self._output_spec

        def dataset_spec(self):
            if self._dataset_spec is None:
                self._dataset_spec = dataset_spec_to_input_spec(self.get_dataset('train'))
            return self._dataset_spec

        @property
        def loss(self):
            return self._loss

        @property
        def metrics(self):
            return list(self._metrics)

        @abc.abstractmethod
        def _get_base_dataset(self, split):
            raise NotImplementedError

        def get_dataset(self, split, batch_size=None, repeat=False, prefetch=True):
            dataset = self._get_base_dataset(split)
            return self.data_pipeline(
                dataset, split, batch_size, repeat=repeat, prefetch=prefetch)

        def data_pipeline(
            self, dataset, split, batch_size, repeat, shuffle=None,
            prefetch=True):
            map_fn = self._map_fn
            if isinstance(map_fn, dict):
                map_fn = map_fn[split]

            if shuffle or (shuffle is None and split == 'train'):
                dataset = dataset.shuffle(self._shuffle_buffer)

            if map_fn is not None:
                dataset = dataset.map(
                    map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            if self._use_inverse_freq_weights:
                dataset = with_inverse_freq_weights(dataset, self.class_counts)

            if repeat:
                dataset = dataset.repeat()

            if batch_size is not None:
                dataset = dataset.batch(batch_size)
            if prefetch:
                dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
            return dataset

        @property
        def class_counts(self):
            if self._class_counts is None:
                with tf.Graph().as_default():
                    u_temp = self._use_inverse_freq_weights
                    self._use_inverse_freq_weights = False
                    ds = self.get_dataset('train', batch_size=None)
                    self._use_inverse_freq_weights = u_temp
                    logging.info('Computing class counts...')

                    def reduce_func(_counts, args):
                        false_counts, true_counts = _counts
                        label = args[1]
                        false_counts, true_counts = tf.cond(
                            label,
                            lambda: (false_counts, true_counts + 1),
                            lambda: (false_counts + 1, true_counts))
                        return false_counts, true_counts

                    counts = ds.reduce(
                        (tf.constant(0, dtype=tf.int32),) * 2, reduce_func)
                    with tf.compat.v1.Session() as sess:
                        counts = sess.run(counts)
                    self._class_counts = counts
                    logging.info('Class counts computed')
            return self._class_counts


    class TfdsProblem(BaseProblem):
        """`BaseProblem` implementation based on `tfds.core.DatasetBuilder`s."""

        def __init__(self, builder, loss, metrics=None, as_supervised=True,
                     output_spec=None, map_fn=None, shuffle_buffer=1024,
                     use_inverse_freq_weights=False,
                     class_counts=None):
            """
            Args:
                builder: `tfds.core.DatasetBuilder` instance.
                rest: see `ml_glaucoma.problems.BaseProblem`
            """
            if map_fn is not None:
                assert (callable(map_fn) or
                        isinstance(map_fn, dict) and
                        all(v is None or callable(v) for v in map_fn.values()))
            self._builder = builder
            self._output_spec = output_spec
            self._as_supervised = as_supervised
            super(TfdsProblem, self).__init__(
                map_fn=map_fn, use_inverse_freq_weights=use_inverse_freq_weights,
                class_counts=class_counts, shuffle_buffer=shuffle_buffer,
                loss=loss, metrics=metrics)

        def _supervised_feature(self, index):
            info = self.builder.info
            keys = info.supervised_keys
            if keys is None:
                return None
            else:
                return info.features[keys[index]]

        def output_spec(self):
            if self._output_spec is not None:
                return self._output_spec

            # attempt to handle supervised problems by default
            feature = self._supervised_feature(1)
            num_classes = getattr(feature, 'num_classes', None)
            if num_classes is not None:
                # categorical classification
                self._output_spec = InputSpec(
                    shape=(num_classes,), dtype=tf.float32)
            elif feature.dtype.is_bool:
                # binary classification
                self._output_spec = InputSpec(
                    shape=(), dtype=tf.float32)
            else:
                return super(TfdsProblem, self).output_spec()
            return self._output_spec

        @property
        def builder(self):
            return self._builder

        def _get_base_dataset(self, split):
            return self.builder.as_dataset(
                batch_size=None,
                split=self._split(split),
                as_supervised=self._as_supervised,
                shuffle_files=True)

        def examples_per_epoch(self, split='train'):
            return _examples_per_epoch(self.builder, self._split(split))

        def _split(self, split):
            """
            Wrapper function that allows re-interpretation of splits.

            For example, a tfds_builder might only provide a train/test split, in
            which case we might redefine the train/val split as a 90%/10% split
            of the original train split.

            ```python
            if split == 'train':
                return tfds.Split.TRAIN.subsplit(tfds.perfect[:90])
            elif split == 'validation':
                return tfds.Split.TRAIN.subsplit(tfds.split[90:])
            elif split == 'test':
                return tfds.Split.TEST
            else:
                raise ValueError("Unrecognized split '{:s}'".format(split))
            ```
            """
            if split == 'validation' and split not in self.builder.info.splits:
                # hacky fallback
                split = 'test'
            return split


    def with_inverse_freq_weights(dataset, counts):
        """
        Add weights to a classification dataset based on class frequencies.

        Args:
            dataset: `tf.data.Dataset` instance with (inputs, labels) as outputs
            counts: list/tuple of ints giving class frequencies

        Returns:
            `tf.data.Dataset` instance with (inputs, labels, weights), where
            weights corresponds to a scaled reciprocal of counts.
        """
        counts = tf.constant(counts, tf.float32)
        class_weights = tf.reduce_mean(counts, keepdims=True) / counts

        def map_fn(inputs, labels):
            weights = class_weights[tf.cast(labels, tf.int32)]
            return inputs, labels, weights

        return dataset.map(
            map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


    class TfdsMultiProblem(BaseProblem):
        """Problem based on multiple `tfds.DatasetBuilder`s."""

        def __init__(self, builders, *args, **kwargs):
            """
            Args:
                builders: iterable of `tfds.core.DatasetBuilder`s
                args/kwargs: see `ml_glaucoma.problems.BaseProblem`.
            """
            self._builders = tuple(builders)
            super(TfdsMultiProblem, self).__init__(*args, **kwargs)

        def _get_base_dataset(self, split):
            datasets = [
                b.as_dataset(split=split, batch_size=1, as_supervised=True)
                for b in self._builders]
            weights = [
                _examples_per_epoch(b, split) for b in self._builders]
            total = sum(weights)
            weights = [w / total for w in weights]
            if split == 'train':
                datasets = [d.shuffle(self._shuffle_buffer) for d in datasets]
                dataset = tf.data.experimental.sample_from_datasets(
                    datasets, weights=weights)
            else:
                # No need to shuffle
                dataset = datasets[0]
                for d in datasets[1:]:
                    dataset = dataset.concatenate(d)
            return dataset

        def get_dataset(self, split, batch_size=None, repeat=False, prefetch=True):
            dataset = self._get_base_dataset(split)
            return self.data_pipeline(
                dataset, split, batch_size, repeat=repeat, prefetch=prefetch,
                shuffle=False  # shuffling occurs in train each sampled dataset
            )

        def examples_per_epoch(self, split=tfds.Split.TRAIN):
            return sum(_examples_per_epoch(b, split) for b in self._builders)

        def input_spec(self):
            return self.dataset_spec()[0]

        def target_spec(self):
            """`target` means label."""
            return self.dataset_spec()[1]


    def preprocess_example(image, labels,
                           pad_to_square=False,
                           resolution=None,
                           use_rgb=True,  # grayscale if False
                           maybe_horizontal_flip=False,
                           maybe_vertical_flip=False,
                           per_image_standardization=True,
                           labels_are_images=False):
        """Preprocessing function for optional flipping/standardization."""

        def maybe_apply(img, _labels, fn, apply_to_labels, prob=0.5):
            apply = tf.random.uniform((), dtype=tf.float32) < prob
            if apply_to_labels:
                return tf.cond(
                    apply,
                    lambda: (fn(img), fn(_labels)),
                    lambda: (img, _labels))
            else:
                return tf.cond(apply, lambda: fn(img), lambda: img), _labels

        if maybe_horizontal_flip:
            image, labels = maybe_apply(
                image, labels, tf.image.flip_left_right, labels_are_images)
        if maybe_vertical_flip:
            image, labels = maybe_apply(
                image, labels, tf.image.flip_up_down, labels_are_images)
        if pad_to_square:
            input_res = tf.shape(image)[-3:-1]
            max_dim = tf.reduce_max(input_res)
            pad_total = max_dim - input_res
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            num_batch_dims = image.shape.ndims - 3  # probably zero
            pad_left = tf.pad(pad_left, [[num_batch_dims, 1]])
            pad_right = tf.pad(pad_right, [[num_batch_dims, 1]])
            paddings = tf.stack((pad_left, pad_right), axis=1)
            image = tf.pad(image, paddings)
            if labels_are_images:
                labels = tf.pad(labels, paddings)
        if resolution is not None:
            def resize(image):
                image = tf.expand_dims(image, axis=0)
                image = tf.image.resize_area(image, resolution, align_corners=True)
                image = tf.squeeze(image, axis=0)
                return image

            image = resize(image)
            if labels_are_images:
                labels = resize(image)

        if not use_rgb:
            image = tf.reduce_mean(image, axis=-1, keepdims=True)
        # image = tf.cast(image, tf.float32)
        image = tf.image.convert_image_dtype(image, tf.float32)
        if per_image_standardization:
            image = tf.image.per_image_standardization(image)
        return image, labels
elif environ['TORCH']:
    def dataset_spec_to_input_spec(*args, **kwargs):
        raise NotImplementedError()


    def preprocess_example(*args, **kwargs):
        raise NotImplementedError()


    def BaseProblem(*args, **kwargs):
        raise NotImplementedError()


    def TfdsProblem(*args, **kwargs):
        raise NotImplementedError()


    def TfdsMultiProblem(*args, **kwargs):
        raise NotImplementedError()


    def with_inverse_freq_weights(*args, **kwargs):
        raise NotImplementedError()
else:
    def dataset_spec_to_input_spec(*args, **kwargs):
        raise NotImplementedError()


    def preprocess_example(*args, **kwargs):
        raise NotImplementedError()


    def BaseProblem(*args, **kwargs):
        raise NotImplementedError()


    def TfdsProblem(*args, **kwargs):
        raise NotImplementedError()


    def TfdsMultiProblem(*args, **kwargs):
        raise NotImplementedError()


    def with_inverse_freq_weights(*args, **kwargs):
        raise NotImplementedError()
