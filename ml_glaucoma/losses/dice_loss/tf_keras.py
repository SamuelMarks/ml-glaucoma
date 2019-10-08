def DiceLoss(y_true, y_pred):
    from tensorflow.keras.backend import binary_crossentropy
    import tensorflow.keras.backend as K

    def dice_loss(y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = y_true_f * y_pred_f
        score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1. - score

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
