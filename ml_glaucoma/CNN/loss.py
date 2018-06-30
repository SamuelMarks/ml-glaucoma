from itertools import product
from platform import python_version_tuple

from keras import backend as K

if python_version_tuple()[0] == '3':
    xrange = range


# From: https://github.com/keras-team/keras/issues/6218
def w_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        nb_cl = len(weights)
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=1, keepdims=True)
        y_pred_max_mat = K.equal(y_pred, y_pred_max)
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t]
        return K.categorical_crossentropy(y_pred, y_true) * final_mask

    return loss
