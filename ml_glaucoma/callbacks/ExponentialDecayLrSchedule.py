def exponential_decay_lr_schedule(lr0, factor):
    """
    lambda epoch: lr0 * (factor ** epoch)

    The returned callback can be used in
    `tf.keras.callbacks.LearningRateScheduler`.
    """
    return lambda epoch: lr0 * (factor ** epoch)
