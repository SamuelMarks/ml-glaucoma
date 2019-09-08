import numpy as np


def SoftAUC(y_true, y_pred):
    import tensorflow.keras.backend as K

    # Extract 1s
    pos_pred_vr = y_pred[y_true.nonzero()]

    # Extract zeroes
    neg_pred_vr = y_pred[np.nonzero(y_true == 0)]

    # Broadcast the subtraction to give a matrix of differences  between pairs of observations
    pred_diffs_vr = np.expand_dims(pos_pred_vr, axis=1) - np.expand_dims(neg_pred_vr, axis=0)

    # Get sigmoid of each pair
    stats = K.sigmoid(pred_diffs_vr * 2)

    # Take average and reverse sign
    return 1 - K.mean(stats, axis=-1)
