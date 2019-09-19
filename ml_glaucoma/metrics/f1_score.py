import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.backend import get_session


# Adapted from Vinicius comment in https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/73929
class F1Metric(tf.keras.layers.Layer):
    def __init__(self, num_classes, threshold, **kwargs):
        super(F1Metric, self).__init__(name='f1', **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        self.stateful = True

        self.tp_mac = None
        self.fp_mac = None
        self.fn_mac = None
        self.local_variables = None

    def reset_states(self):
        get_session().run(tf.variables_initializer(self.local_variables))

    @staticmethod
    def metric_variable(shape, dtype, validate_shape=True, name=None):
        return tf.Variable(
            np.zeros(shape),
            dtype=dtype,
            trainable=False,
            collections=[tf.GraphKeys.LOCAL_VARIABLES],
            validate_shape=validate_shape,
            name=name,
        )

    def streaming_counts(self, y_true, y_pred):
        self.tp_mac = Metrics.metric_variable(
            shape=[self.num_classes], dtype=tf.int64, validate_shape=False, name='tp_mac'
        )
        self.fp_mac = Metrics.metric_variable(
            shape=[self.num_classes], dtype=tf.int64, validate_shape=False, name='fp_mac'
        )
        self.fn_mac = Metrics.metric_variable(
            shape=[self.num_classes], dtype=tf.int64, validate_shape=False, name='fn_mac'
        )

        up_tp_mac = tf.assign_add(self.tp_mac, tf.count_nonzero(y_pred * y_true, axis=0))
        self.add_update(up_tp_mac)
        up_fp_mac = tf.assign_add(self.fp_mac, tf.count_nonzero(y_pred * (y_true - 1), axis=0))
        self.add_update(up_fp_mac)
        up_fn_mac = tf.assign_add(self.fn_mac, tf.count_nonzero((y_pred - 1) * y_true, axis=0))
        self.add_update(up_fn_mac)

        self.local_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)

    def __call__(self, y_true_and_y_pred, **kwargs):
        y_true, y_pred = y_true_and_y_pred
        rounded_pred = K.cast(K.greater_equal(y_pred, self.threshold), 'float32')
        self.streaming_counts(y_true, rounded_pred)
        prec_mac = self.tp_mac / (self.tp_mac + self.fp_mac)
        rec_mac = self.tp_mac / (self.tp_mac + self.fn_mac)
        f1_mac = 2 * prec_mac * rec_mac / (prec_mac + rec_mac)
        f1_mac = tf.reduce_mean(tf.where(tf.is_nan(f1_mac), tf.zeros_like(f1_mac), f1_mac))
        return f1_mac
