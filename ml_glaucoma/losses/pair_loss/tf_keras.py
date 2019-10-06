import tensorflow as tf


def PairLoss(y_true, y_pred):
    import tensorflow.keras.backend as K

    y_true = tf.cast(y_true, tf.int32)
    parts = tf.dynamic_partition(y_pred, y_true, 2)
    y_pos = parts[1]
    y_neg = parts[0]
    y_pos = tf.expand_dims(y_pos, 0)
    y_neg = tf.expand_dims(y_neg, -1)
    out = K.sigmoid(y_neg - y_pos)
    return K.mean(out)
