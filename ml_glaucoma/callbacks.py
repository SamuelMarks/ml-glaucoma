from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow as tf
import gin.tf
import os
from ml_glaucoma.tf_compat import is_v1


GinConfigSaverCallback = gin.config.external_configurable(
    gin.tf.GinConfigSaverCallback)


@gin.configurable
class CheckpointManagerCallback(tf.keras.callbacks.Callback):
    """
    Callback wraping `tf.train.CheckpointManager`.

    Restores previous checkpoint `on_train_begin` and saves every `period`
    epochs and optional at the end of training.
    """
    def __init__(
            self, model_dir, period, save_on_train_end=True,
            **manager_kwargs):
        self._model_dir = model_dir
        self._period = period
        self._save_on_train_end = save_on_train_end
        self._manager_kwargs = manager_kwargs
        self._restored = False
        self._manager = None
        self._checkpoint = None
        self._epoch_count = None
        self._last_save = None

    @property
    def manager(self):
        if self._manager is None:
            self._manager = tf.train.CheckpointManager(
                self.checkpoint, self._model_dir, **self._manager_kwargs)
        return self._manager

    @property
    def checkpoint(self):
        if self._checkpoint is None:
            self._checkpoint = tf.train.Checkpoint(model=self.model)
        return self._checkpoint

    def _on_begin(self):
        if not self._restored:
            self.restore()

    def restore(self, save_path=None):
        if save_path is None:
            save_path = self.manager.latest_checkpoint
        self.checkpoint.restore(save_path)
        self._restored = True

    def on_train_begin(self, logs=None):
        self._on_begin()

    def on_test_begin(self, logs=None):
        self._on_begin()

    def on_predict_begin(self, logs=None):
        self._on_begin()

    def on_epoch_end(self, epoch, logs=None):
        self._epoch_count = epoch + 1
        print(self._epoch_count % self._period)
        if self._epoch_count % self._period == 0:
            self._save()

    def on_train_end(self, logs=None):
        if self._save_on_train_end:
            self._save()

    def _save(self):
        if self._epoch_count is None:
            return
        if self._last_save != self._epoch_count:
            self.manager.save(self._epoch_count)
            self._last_save = self._epoch_count


@gin.configurable
def exponential_decay_lr_schedule(lr0, factor):
    def f(epoch):
        return lr0 * (factor ** epoch)
    return f


def get_callbacks(
        model,
        batch_size,
        callbacks=None,
        checkpoint_freq=5,
        summary_freq=10,
        save_config=True,
        model_dir=None,
        train_steps_per_epoch=None,
        val_steps_per_epoch=None,
        lr_schedule=None,
    ):
    if callbacks is None:
        callbacks = []
    else:
        callbacks = list(callbacks)

    initial_epoch = None
    if checkpoint_freq is not None:
        # Issues in the following related to ListWrappers/Checkpointable?
        saver_callback = CheckpointManagerCallback(
            model_dir, period=checkpoint_freq, max_to_keep=5)
        saver_callback.set_model(model)
        saver_callback.restore()
        chkpt = saver_callback.manager.latest_checkpoint
        if chkpt is not None:
            if is_v1:
                for substr in chkpt.split('.')[-1::-1]:
                    try:
                        last_step = int(substr)
                        assert(last_step % train_steps_per_epoch == 0)
                        initial_epoch = last_step // train_steps_per_epoch
                        break
                    except Exception:
                        pass
                else:
                    raise RuntimeError(
                        'Unrecognized checkpoint prefix %s' % chkpt)
            else:
                initial_epoch = int(
                    chkpt.split('/')[-1].split('.')[0].split('-')[-1])

        callbacks.append(saver_callback)

    if initial_epoch is None:
        initial_epoch = 0

    if summary_freq:
        kwargs = dict(
            write_graph=False, log_dir=model_dir, update_freq=summary_freq)
        tb_callback = tf.keras.callbacks.TensorBoard(**kwargs)

        # These hacks involve private members - will probably break
        if train_steps_per_epoch is not None and initial_epoch > 0:
            initial_train_steps = \
                initial_epoch*train_steps_per_epoch
            tb_callback._total_batches_seen = initial_train_steps
            tb_callback._samples_seen = initial_train_steps*batch_size
        if val_steps_per_epoch is not None and initial_epoch > 0:
            initial_val_steps = initial_epoch*val_steps_per_epoch
            tb_callback._total_val_batches_seen = initial_val_steps

        callbacks.append(tb_callback)

    if lr_schedule is not None:
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_schedule))

    if save_config:
        callbacks.append(GinConfigSaverCallback(model_dir))

    return callbacks, initial_epoch
