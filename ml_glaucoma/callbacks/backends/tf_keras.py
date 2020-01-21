import tensorflow as tf

from ml_glaucoma.callbacks.DropWorseModels.tf_keras import DropWorseModels
from ml_glaucoma.callbacks.loading_model_checkpoint.tf_keras import LoadingModelCheckpoint
from ml_glaucoma.tf_compat import is_tf_v1


def get_callbacks(
    batch_size,
    checkpoint_freq='epoch',
    summary_freq=10,
    model_dir=None,
    train_steps_per_epoch=None,
    val_steps_per_epoch=None,
    lr_schedule=None,
    tensorboard_log_dir=None,
    write_images=False,
):
    """
    Get common callbacks used in training.

    Args:
        batch_size: size of each batch - used to correct tensorboard initial
            step.
        checkpoint_freq: if not None, adds a `LoadingModelCheckpoint` which
            extends `ModelCheckpoint` to restore weights on fit/evaluate start
            and saves at this epoch frequency.
        summary_freq: if given, adds a `TensorBoard` callback that logs at this
            batch frequency.
        model_dir: directory in which to save weights
        train_steps_per_epoch: number of training steps per epoch. Necessary
            for initializing `TensorBoard` correctly when resuming training.
        val_steps_per_epoch: number of validation steps per epoch.
        lr_schedule: if provided, adds a `LearningRateScheduler` with this
            schedule.
        tensorboard_log_dir: if given, logs are written here. It not,
            `model_dir` is used
        write_images: passed to `TensorBoard`

    Returns:
        (callbacks, initial_epoch), where callbacks is a list of
        `tf.keras.callbacks.Callback` and `initial_epoch` corresponds to the
        epoch count of the weights loaded.
    """
    callbacks = []

    initial_epoch = 0
    if checkpoint_freq is not None:
        saver_callback = LoadingModelCheckpoint(
            model_dir, save_freq=checkpoint_freq, save_best_only=False)
        latest_checkpoint = saver_callback.latest_checkpoint
        if latest_checkpoint is not None:
            initial_epoch = LoadingModelCheckpoint.filename_epoch(latest_checkpoint)
        callbacks.append(saver_callback)

    if summary_freq:
        tb_callback = tf.keras.callbacks.TensorBoard(
            write_graph=False,
            log_dir=tensorboard_log_dir or model_dir,
            update_freq=summary_freq,
            write_images=write_images)

        # These hacks involve private members - will probably break
        if train_steps_per_epoch is not None and initial_epoch > 0:
            initial_train_steps = \
                initial_epoch * train_steps_per_epoch
            tb_callback._total_batches_seen = initial_train_steps
            # v1 a sample is a batch, where as in v2 a sample is an element
            if is_tf_v1:
                tb_callback._samples_seen = initial_train_steps
            else:
                tb_callback._samples_seen = initial_train_steps * batch_size
        if val_steps_per_epoch is not None and initial_epoch > 0:
            initial_val_steps = initial_epoch * val_steps_per_epoch
            tb_callback._total_val_batches_seen = initial_val_steps

        callbacks.append(tb_callback)

        callbacks.append(DropWorseModels(save_best_only=True))

    if lr_schedule is not None:
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_schedule))

    return callbacks, initial_epoch
