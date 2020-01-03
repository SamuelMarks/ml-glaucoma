import tensorflow as tf


class AUCall(tf.keras.metrics.AUC):
    """
    tf.keras.metrics.AUC but also emit TP, FP, TN, FN
    """

    def __init__(self, writer,
                 num_thresholds=200,
                 curve='ROC',
                 summation_method='interpolation',
                 name='AUCall',
                 dtype=None,
                 thresholds=None):
        super(AUCall, self).__init__(num_thresholds=num_thresholds,
                                     curve=curve,
                                     summation_method=summation_method,
                                     name=name,
                                     dtype=dtype,
                                     thresholds=thresholds)
        self.writer = writer

    def confusion_logger(self):
        with tf.compat.v1.Graph().as_default():
            step = tf.Variable(0, dtype=tf.int64)
            step_update = step.assign_add(1)
            with self.writer.as_default(), tf.name_scope('auc'):
                tf.summary.scalar('tp', data=self.true_positives, description='true_positives')
                tf.summary.scalar('fp', data=self.false_positives, description='false_positives')
                tf.summary.scalar('tn', data=self.true_negatives, description='true_negatives')
                tf.summary.scalar('fn', data=self.false_negatives, description='false_negatives')
                all_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
            writer_flush = self.writer.flush()

            # TODO: Use global session?
            sess = tf.compat.v1.Session()
            sess.run([self.writer.init(), step.initializer])
            sess.run(all_summary_ops)
            sess.run(step_update)
            sess.run(writer_flush)

    def result(self):
        self.confusion_logger()
        return super(AUCall, self).result()
