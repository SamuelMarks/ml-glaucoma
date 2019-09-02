from platform import python_version_tuple

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.engine import Layer
from sklearn.metrics import precision_recall_fscore_support, fbeta_score

from ml_glaucoma.CNN.helpers import output_sensitivity_specificity

if python_version_tuple()[0] == '3':
    xrange = range

'''
# Adapted from: https://github.com/keras-team/keras/issues/3358#issuecomment-334460423
class SensitivitySpecificityCallback(TensorBoard):
    """Sets the self.validation_data property for use with SensitivitySpecificityCallback callback."""

    def __init__(self, batch_gen, **kwargs):
        super(SensitivitySpecificityCallback, self).__init__(**kwargs)
        self.batch_gen = batch_gen  # The generator.

    def on_epoch_end(self, epoch, logs=None):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in xrange(self.nb_steps):

            if imgs is None and tags is None:
                imgs = np.zeros(((self.nb_steps * ib.shape[0],) + ib.shape[1:]), dtype=np.float32)
                tags = np.zeros(((self.nb_steps * tb.shape[0],) + tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        if epoch:
            print('SensitivitySpecificityCallback::self.validation_data:', self.validation_data, ';')
            print('SensitivitySpecificityCallback::self:', self, ';')
            for k in dir(self):
                print('SensitivitySpecificityCallback::self.{}:'.format(k), getattr(self, k), ';')

            for k in dir(self.model):
                print('SensitivitySpecificityCallback::self.model.{}:'.format(k), getattr(self.model, k), ';')
            # `self.model.validation_data` ?
            x_test, y_test = self.validation_data[0], self.validation_data[1]
            predictions = self.model.predict(x_test)
            output_sensitivity_specificity(epoch, predictions, y_test)
        return super(SensitivitySpecificityCallback, self).on_epoch_end(epoch, logs)
'''


class SensitivitySpecificityCallback(Callback):
    validation_data_explicit = None
    class_mode = None

    def __init__(self, validation_data, class_mode):
        super(SensitivitySpecificityCallback, self).__init__()
        if self.validation_data is None:
            self.validation_data_explicit = validation_data
        self.class_mode = class_mode

    def on_epoch_end(self, epoch, logs=None):
        if epoch:
            if self.validation_data is None:
                print('self.validation_data is None')
                self.validation_data_explicit = self.validation_data

            x_test, y_test = self.validation_data[0], self.validation_data[1]
            predictions = self.model.predict(x_test)

            import numpy as np

            print('x_test =', x_test, ';')
            print('y_test =', y_test, ';')
            print('predictions =', predictions, ';')

            np.save('/tmp/x_test', x_test)
            np.save('/tmp/y_test', y_test)
            np.save('/tmp/predictions', predictions)

            output_sensitivity_specificity(epoch, predictions, y_test,
                                           class_mode=self.class_mode)


# from: https://stackoverflow.com/a/48720556
def reweigh(y_true, y_pred, tp_weight=0.2, tn_weight=0.2, fp_weight=1.2, fn_weight=1.2):
    # Get predictions
    y_pred_classes = K.greater_equal(y_pred, 0.5)
    y_pred_classes_float = K.cast(y_pred_classes, K.floatx())

    # Get misclassified examples
    wrongly_classified = K.not_equal(y_true, y_pred_classes_float)
    wrongly_classified_float = K.cast(wrongly_classified, K.floatx())

    # Get correctly classified examples
    correctly_classified = K.equal(y_true, y_pred_classes_float)
    correctly_classified_float = K.cast(wrongly_classified, K.floatx())

    # Get tp, fp, tn, fn
    tp = correctly_classified_float * y_true
    tn = correctly_classified_float * (1 - y_true)
    fp = wrongly_classified_float * y_true
    fn = wrongly_classified_float * (1 - y_true)

    # Get weights
    weight_tensor = tp_weight * tp + fp_weight * fp + tn_weight * tn + fn_weight * fn

    loss = K.binary_crossentropy(y_true, y_pred)
    weighted_loss = loss * weight_tensor
    return weighted_loss


def f_score_obj(y_true, y_pred):
    y_true = K.eval(y_true)
    y_pred = K.eval(y_pred)
    _precision, _recall, f_score, _support = precision_recall_fscore_support(y_true, y_pred)
    return K.variable(1. - f_score[1])


def precision(y_true, y_pred):
    """ Calculates the precision """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    _precision = true_positives / (predicted_positives + K.epsilon())
    return _precision


def recall(y_true, y_pred):
    """ Calculates the recall """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    _recall = true_positives / (possible_positives + K.epsilon())
    return _recall


def fmeasure(y_true, y_pred):
    """ Calculates the f-measure, the harmonic mean of precision and recall. """
    return fbeta_score(y_true, y_pred, beta=1)


def binary_segmentation_recall(y_true, y_pred):
    # We assume y_true and y_pred have shape [?, W, H, 1]
    # This is a segmentation loss, for classification it doesn't really make sense.
    tp_ish = K.hard_sigmoid(100. * (y_pred - 0.5))
    approx_recall = K.sum(tp_ish, [1, 2, 3]) / K.sum(y_true, axis=[1, 2, 3])
    return approx_recall


class BinaryTruePositives(Layer):
    """Stateful Metric to count the total true positives over all batches.
        Assumes predictions and targets of shape `(samples, 1)`.
        # Arguments
            name: String, name for the metric.
    """

    def __init__(self, name='true_positives', **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.stateful = True
        self.true_positives = K.variable(value=0, dtype='int32')

    def reset_states(self):
        K.set_value(self.true_positives, 0)

    def __call__(self, y_true, y_pred):
        """Computes the number of true positives in a batch.
        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions
        # Returns
            The total number of true positives seen this epoch at the
                completion of the batch.
        """
        y_true = K.cast(y_true, 'int32')
        y_pred = K.cast(K.round(y_pred), 'int32')
        correct_preds = K.cast(K.equal(y_pred, y_true), 'int32')
        true_pos = K.cast(K.sum(correct_preds * y_true), 'int32')
        current_true_pos = self.true_positives * 1
        self.add_update(K.update_add(self.true_positives, true_pos), inputs=[y_true, y_pred])

        return current_true_pos + true_pos


# custom metrics for categorical
def specificity(y_true, y_pred):
    return K.cast(K.all((
        K.equal(K.argmax(y_true, axis=-1), 1),
        K.equal(K.argmax(y_pred, axis=-1), 1)
    ), axis=1), K.floatx())


def sensitivity(y_true, y_pred):
    return K.cast(K.all((
        K.equal(K.argmax(y_true, axis=-1), 2),
        K.equal(K.argmax(y_pred, axis=-1), 2)
    ), axis=1), K.floatx())


def specificity_at_sensitivity(sensitivity_value, **kwargs):
    def metric(labels, predictions):
        # any tensorflow metric
        value, update_op = tf.metrics.specificity_at_sensitivity(labels, predictions, sensitivity_value, **kwargs)

        # find all variables created for this metric
        metric_vars = (i for i in tf.local_variables() if 'specificity_at_sensitivity' in i.name.split('/')[2])

        # Add metric variables to GLOBAL_VARIABLES collection.
        # They will be initialized for new session.
        for v in metric_vars:
            tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

        # force to update metric values
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
            return value

    return metric


# https://github.com/keras-team/keras/blob/e583c566f0fd9bf5d39a8b081a872f7d32e02480/keras/metrics.py

class Recall(Layer):
    """Compute recall over all batches.
    # Arguments
        name: String, name for the metric.
        class_ind: Integer, class index.
    """

    def __init__(self, name='recall', class_ind=1):
        super(Recall, self).__init__(name=name)
        self.true_positives = K.variable(value=0, dtype='float32')
        self.total_positives = K.variable(value=0, dtype='float32')
        self.class_ind = class_ind

    def reset_states(self):
        K.set_value(self.true_positives, 0.0)
        K.set_value(self.total_positives, 0.0)

    def __call__(self, y_true, y_pred):
        """Update recall computation.
        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions
        # Returns
            Overall recall for the epoch at the completion of the batch.
        """
        # Batch
        y_true, y_pred = _slice_by_class(y_true, y_pred, self.class_ind)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        total_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        # Current
        current_true_positives = self.true_positives * 1
        current_total_positives = self.total_positives * 1

        # Updates
        updates = [K.update_add(self.true_positives, true_positives),
                   K.update_add(self.total_positives, total_positives)]
        self.add_update(updates, inputs=[y_true, y_pred])

        # Compute recall
        return (current_true_positives + true_positives) / \
               (current_total_positives + total_positives + K.epsilon())


class Precision(Layer):
    """Compute precision over all batches.
    # Arguments
        name: String, name for the metric.
        class_ind: Integer, class index.
    """

    def __init__(self, name='precision', class_ind=1):
        super(Precision, self).__init__(name=name)
        self.true_positives = K.variable(value=0, dtype='float32')
        self.pred_positives = K.variable(value=0, dtype='float32')
        self.class_ind = class_ind

    def reset_states(self):
        K.set_value(self.true_positives, 0.0)
        K.set_value(self.pred_positives, 0.0)

    def __call__(self, y_true, y_pred):
        """Update precision computation.
        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions
        # Returns
            Overall precision for the epoch at the completion of the batch.
        """
        # Batch
        y_true, y_pred = _slice_by_class(y_true, y_pred, self.class_ind)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        pred_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        # Current
        current_true_positives = self.true_positives * 1
        current_pred_positives = self.pred_positives * 1

        # Updates
        updates = [K.update_add(self.true_positives, true_positives),
                   K.update_add(self.pred_positives, pred_positives)]
        self.add_update(updates, inputs=[y_true, y_pred])

        # Compute recall
        return (current_true_positives + true_positives) / \
               (current_pred_positives + pred_positives + K.epsilon())


def _slice_by_class(y_true, y_pred, class_ind):
    """ Slice the batch predictions and labels with respect to a given class
    that is encoded by a categorical or binary label.
    #  Arguments:
        y_true: Tensor, batch_wise labels.
        y_pred: Tensor, batch_wise predictions.
        class_ind: Integer, class index.
    # Returns:
        y_slice_true: Tensor, batch_wise label slice.
        y_slice_pred: Tensor,  batch_wise predictions, slice.
    """
    # Binary encoded
    if y_pred.shape[-1] == 1:
        y_slice_true, y_slice_pred = y_true, y_pred
    # Categorical encoded
    else:
        y_slice_true, y_slice_pred = y_true[..., class_ind], y_pred[..., class_ind]
    return y_slice_true, y_slice_pred
