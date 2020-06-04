import tensorflow as tf


def Kappa(y_pred, y_true, y_pow=2, eps=1e-10, N=5, batch_size=1, name='kappa'):
    """A continuous differentiable approximation of discrete kappa loss.
        Args:
            y_pred: 2D tensor or array, [batch_size, num_classes]
            y_true: 2D tensor or array,[batch_size, num_classes]
            y_pow: int,  e.g. y_pow=2
            N: typically num_classes of the model
            batch_size: batch_size of the training or validation ops
            eps: a float, prevents divide by zero
            name: Optional scope/name for op_scope.
        Returns:
            A tensor with the kappa loss."""

    with tf.compat.v1.name_scope(name):
        y_true = tf.cast(y_true, dtype=tf.float32)
        repeat_op = tf.cast(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]), dtype=tf.float32)
        repeat_op_sq = tf.square((repeat_op - tf.transpose(a=repeat_op)))
        weights = repeat_op_sq / tf.cast((N - 1) ** 2, dtype=tf.float32)

        pred_ = y_pred ** y_pow
        try:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(input_tensor=pred_, axis=1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(input_tensor=pred_, axis=1), [batch_size, 1]))

        hist_rater_a = tf.reduce_sum(input_tensor=pred_norm, axis=0)
        hist_rater_b = tf.reduce_sum(input_tensor=y_true, axis=0)

        conf_mat = tf.matmul(tf.transpose(a=pred_norm), y_true)

        nom = tf.reduce_sum(input_tensor=weights * conf_mat)
        denom = tf.reduce_sum(input_tensor=weights * tf.matmul(
            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /
                                           tf.cast(batch_size, dtype=tf.float32))

        return nom / (denom + eps)
