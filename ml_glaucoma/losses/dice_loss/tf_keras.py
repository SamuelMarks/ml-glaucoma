def DiceLoss(y_true, y_pred):
    import tensorflow.keras.backend as K
    from tensorflow.keras.backend import binary_crossentropy

    def dice_loss(y_true, y_pred):
        smooth = 1.0
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = y_true_f * y_pred_f
        score = (2.0 * K.sum(intersection) + smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + smooth
        )
        return 1.0 - score

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
