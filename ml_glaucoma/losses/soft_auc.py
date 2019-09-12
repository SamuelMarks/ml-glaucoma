import numpy as np


def SoftAUC(y_true, y_pred):
    import tensorflow as tf
    import tensorflow.keras.backend as K
    # Extract 1s
    # pos_pred_vr = y_pred[y_true.nonzero()]

    # Extract zeroes
    # neg_pred_vr = y_pred[np.nonzero(y_true == 0)]

    # Extract zeros and ones
    if y_true.dtype != tf.bool:
        raise ValueError('y_true must be bool, got {}'.format(y_true.dtype))
    neg_pred_vr, pos_pred_vr = tf.dynamic_partition(
        y_pred, tf.cast(y_true, tf.int32), num_partitions=2)

    # Broadcast the subtraction to give a matrix of differences  between pairs of observations
    pred_diffs_vr = tf.expand_dims(pos_pred_vr, axis=1) - tf.expand_dims(neg_pred_vr, axis=0)

    # Get sigmoid of each pair
    stats = K.sigmoid(pred_diffs_vr * 2)

    # Take average and reverse sign
    return 1 - K.mean(stats)


if __name__ == '__main__':
    import tensorflow as tf
    with tf.device('/cpu:0'):
        y_true = tf.constant([False, False, True, True, False])
        y_pred = tf.constant([0.3, 0.3, 0.6, 0.6, 0.4], dtype=tf.float32)
        loss = SoftAUC(y_true, y_pred)
        if tf.executing_eagerly():
            print(loss.numpy())
        else:
            with tf.compat.v1.Session() as sess:
                print(sess.run(loss))
