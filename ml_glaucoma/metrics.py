from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
K = tf.keras.backend


class BinaryPrecision(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, **kwargs):
        super(BinaryPrecision, self).__init__(**kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(
            'true_positives', shape=(), initializer=tf.zeros_initializer,
            dtype=tf.float32)
        self.false_positives = self.add_weight(
            'false_positives', shape=(), initializer=tf.zeros_initializer,
            dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.greater(tf.squeeze(y_pred, axis=-1), self.threshold)
        y_true = tf.squeeze(y_true, axis=-1)
        assert(y_pred.shape.ndims == y_true.shape.ndims)
        if not y_true.dtype.is_bool:
            y_true = tf.cast(y_true, tf.bool)
        true_pos = tf.logical_and(y_pred, y_true)
        false_pos = tf.logical_and(y_pred, tf.logical_not(y_true))
        if sample_weight is None:
            true_pos = tf.cast(tf.math.count_nonzero(true_pos), tf.float32)
            false_pos = tf.cast(tf.math.count_nonzero(false_pos), tf.float32)
        else:
            true_pos = tf.reduce_sum(
                tf.boolean_mask(sample_weight, true_pos))
            false_pos = tf.reduce_sum(
                tf.boolean_mask(sample_weight, false_pos))
        self.true_positives.assign_add(true_pos)
        self.false_positives.assign_add(false_pos)

    def reset_states(self):
        for v in self.true_positives, self.false_positives:
            v.assign(0.)

    def result(self):
        return \
            self.true_positives / (self.true_positives + self.false_positives)

    def get_config(self):
        config = super(BinaryPrecision, self).get_config()
        config['threshold'] = self.threshold
        return config


class BinaryRecall(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, **kwargs):
        super(BinaryRecall, self).__init__(**kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(
            'true_positives', shape=(), initializer=tf.zeros_initializer,
            dtype=tf.float32)
        self.false_negatives = self.add_weight(
            'false_negatives', shape=(), initializer=tf.zeros_initializer,
            dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        tf.print(tf.reduce_max(y_pred), tf.reduce_max(y_true))
        y_pred = tf.greater(tf.squeeze(y_pred, axis=-1), self.threshold)
        y_true = tf.squeeze(y_true, axis=-1)
        assert(y_pred.shape.ndims == y_true.shape.ndims)
        if not y_true.dtype.is_bool:
            y_true = tf.cast(y_true, tf.bool)

        true_pos = tf.logical_and(y_pred, y_true)
        false_neg = tf.logical_and(tf.logical_not(y_pred), y_true)
        if sample_weight is None:
            true_pos = tf.cast(tf.math.count_nonzero(true_pos), tf.float32)
            false_neg = tf.cast(tf.math.count_nonzero(false_neg), tf.float32)
        else:
            true_pos = tf.reduce_sum(
                tf.boolean_mask(sample_weight, true_pos))
            false_neg = tf.reduce_sum(
                tf.boolean_mask(sample_weight, false_neg))
        self.true_positives.assign_add(true_pos)
        self.false_negatives.assign_add(false_neg)

    def reset_states(self):
        for v in self.true_positives, self.false_negatives:
            v.assign(0.)

    def result(self):
        return \
            self.true_positives / (self.true_positives + self.false_negatives)

    def get_config(self):
        config = super(BinaryRecall, self).get_config()
        config['threshold'] = self.threshold
        return config


class BinaryAdapter(tf.keras.metrics.Metric):
    def __init__(self, base_metric, **kwargs):
        base_metric = tf.keras.metrics.get(base_metric)
        self.base_metric = base_metric
        super(BinaryAdapter, self).__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.concat([1. - y_pred, y_pred], axis=-1)
        self.base_metric.update_state(y_true, y_pred, sample_weight)

    def reset_states(self):
        self.base_metric.reset_states()

    def result(self):
        return self.base_metric.result()

    def get_config(self):
        config = super(BinaryAdapter, self).get_config()
        config['base_metric'] = base_metric
        return config
