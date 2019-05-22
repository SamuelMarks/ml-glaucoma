"""
Problems are mostly adapters operating between base datasets and keras models.

While `tensorflow_datasets.DatasetBuilder`s (see `ml_glaucoma.tfds_builders`
for implementations) provides data download, and serialization and meta-data
collection, problems provide a customizable interface for the models to be
trained. They also include metrics, losses and data augmentation preprocessing.

The `Problem` class provdes the general interface, while `TfdsProblem` is a
basic implementation that leverages `tensorflow_datasets.DatasetBuilder`s.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from absl import logging
import six
import tensorflow as tf
import tensorflow_datasets as tfds

InputSpec = tf.keras.layers.InputSpec


def dataset_spec(dataset, has_batch_dim=False):
    if has_batch_dim:
        def f(shape, dtype):
            return InputSpec(shape=shape[1:], dtype=dtype)
    else:
        def f(shape, dtype):
            return InputSpec(shape=shape, dtype=dtype)
    return tf.nest.map_structure(
        f, dataset.output_shapes, dataset.output_types)


class Problem(object):
    @abc.abstractmethod
    def get_dataset(self, split, repeat=False, batch_size=None):
        raise NotImplementedError

    @abc.abstractmethod
    def examples_per_epoch(self, split=tfds.Split.TRAIN):
        raise NotImplementedError

    @abc.abstractproperty
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
        return dataset_spec(self.get_dataset('train'))

    def input_spec(self):
        return self.dataset_spec()[0]

    def target_spec(self):
        """`target` means label."""
        return self.dataset_spec()[1]


def download_and_prepare(builder, download_config=None, download_dir=None):
    if download_config is None:
        download_config = DownloadConfig()
    download_config = download_config
    builder.download_and_prepare(
        download_dir=download_dir, download_config=download_config)


class TfdsProblem(Problem):
    def __init__(
            self, builder, loss, metrics=None, output_spec=None, map_fn=None,
            as_supervised=True, shuffle_buffer=1024,
            use_inverse_freq_weights=False,
            class_counts=None):
        if map_fn is not None:
            assert(callable(map_fn) or
                   isinstance(map_fn, dict) and
                   all(v is None or callable(v) for v in map_fn.values()))
        self._builder = builder
        self._loss = loss
        self._metrics = metrics
        self._output_spec = output_spec
        self._map_fn = map_fn
        self._as_supervised = as_supervised
        if shuffle_buffer is None:
            shuffle_buffer = self.examples_per_epoch('train')
        self._shuffle_buffer = shuffle_buffer
        self._use_inverse_freq_weights = use_inverse_freq_weights
        self._class_counts = None

    @property
    def class_counts(self):
        if self._class_counts is None:
            with tf.Graph().as_default():
                u_temp = self._use_inverse_freq_weights
                self._use_inverse_freq_weights = False
                ds = self.get_dataset('train', batch_size=None)
                self._use_inverse_freq_weights = u_temp
                logging.info('Computing class counts...')
                def reduce_func(counts, args):
                    false_counts, true_counts = counts
                    label = args[1]
                    false_counts, true_counts = tf.cond(
                        label,
                        lambda: (false_counts, true_counts + 1),
                        lambda: (false_counts + 1, true_counts))
                    return (false_counts, true_counts)

                counts = ds.reduce(
                    (tf.constant(0, dtype=tf.int32),)*2, reduce_func)
                with tf.compat.v1.Session() as sess:
                    counts = sess.run(counts)
                self._class_counts = counts
                logging.info('Class counts computed')
        return self._class_counts

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
            return InputSpec(shape=(num_classes,), dtype=tf.float32)
        elif feature.dtype.is_bool:
            # binary classification
            return InputSpec(shape=(), dtype=tf.float32)
        return super(TfdsProblem, self).output_spec()

    @property
    def builder(self):
        return self._builder

    def get_dataset(self, split, batch_size=None, repeat=False, prefetch=True):
        dataset = self.builder.as_dataset(
            batch_size=1, split=self._split(split),
            as_supervised=self._as_supervised, shuffle_files=True)
        return self.data_pipeline(
            dataset, split, batch_size, repeat=repeat, prefetch=prefetch)

    def examples_per_epoch(self, split='train'):
        return int(self.builder.info.splits[self._split(split)].num_examples)

    @property
    def loss(self):
        return self._loss

    @property
    def metrics(self):
        return list(self._metrics)

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
            raise ValueError("Unrecognized split '%s'" % split)
        ```
        """
        if (split == 'validation' and split not in self.builder.info.splits):
            # hacky fallback
            split = 'test'
        return split

    def data_pipeline(self, dataset, split, batch_size, repeat, prefetch=True):
        map_fn = self._map_fn
        if isinstance(map_fn, dict):
            map_fn = map_fn[split]

        if split == 'train':
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


def with_inverse_freq_weights(dataset, counts):
    counts = tf.constant(counts, tf.float32)
    class_weights = tf.reduce_mean(counts, keepdims=True) / counts

    def map_fn(inputs, labels):
        weights = class_weights[tf.cast(labels, tf.int32)]
        return inputs, labels, weights

    return dataset.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def _maybe_apply(image, labels, fn, apply_to_labels, prob=0.5):
    apply = tf.random.uniform((), dtype=tf.float32) < prob
    if apply_to_labels:
        return tf.cond(
            apply, lambda: (fn(image), fn(labels)), lambda: (image, labels))
    else:
        return tf.cond(apply, lambda: fn(image), lambda: image), labels


def preprocess_example(
        image, labels,
        use_rgb=True,  # grayscale if False
        maybe_horizontal_flip=False,
        maybe_vertical_flip=False,
        per_image_standardization=True,
        labels_are_images=False):
    """Preprocessing function for optional flipping/standardization."""
    if maybe_horizontal_flip:
        image, labels = _maybe_apply(
            image, labels, tf.image.flip_left_right, labels_are_images)
    if maybe_vertical_flip:
        image, labels = _maybe_apply(
            image, labels, tf.image.flip_up_down, labels_are_images)
    if not use_rgb:
        image = tf.reduce_mean(image, axis=-1, keepdims=True)
    if per_image_standardization:
        image = tf.image.per_image_standardization(image)
    return image, labels
