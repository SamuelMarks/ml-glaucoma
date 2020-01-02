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

    @tf.function
    def confusion_logger(self):
        # other model code would go here
        with self.writer.as_default(), tf.name_scope('auc'):
            tf.summary.scalar('tp', data=self.true_positives, description='true_positives')
            tf.summary.scalar('fp', data=self.false_positives, description='false_positives')
            tf.summary.scalar('tn', data=self.true_negatives, description='true_negatives')
            tf.summary.scalar('fn', data=self.false_negatives, description='false_negatives')
        self.writer.flush()

    def result(self):
        self.confusion_logger()
        return super(AUCall, self).result()
