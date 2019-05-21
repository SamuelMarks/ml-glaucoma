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


class LoadingModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """ModelCheckpoint modified to automatically restore model."""
    def __init__(self, model_dir, **kwargs):
        self._model_dir = model_dir
        self._filename='model-{epoch:04d}.h5'
        super(LoadingModelCheckpoint, self).__init__(
            filepath=os.path.join(self._model_dir, self._filename), **kwargs)
        self._restored = False

    def restore(self, save_path=None):
        if not self._restored:
            if save_path is None:
                save_path = self.latest_checkpoint
            if save_path is not None:
                self.model.load_weights(save_path)
            self._restored = True

    @property
    def latest_checkpoint(self):
        filenames = tuple(fn for fn in os.listdir(self._model_dir)
                          if fn.startswith('model'))
        if len(filenames) == 0:
            return None
        latest = max(filenames, key=self.filename_epoch)
        return os.path.join(self._model_dir, latest)

    def filename_epoch(self, filename):
        assert(filename.endswith('.h5'))
        return int(filename[-7:-3])

    def on_train_begin(self, logs=None):
        self._on_begin()

    def on_test_begin(self, logs=None):
        self._on_begin()

    def on_predict_begin(self, logs=None):
        self._on_begin()

    def _on_begin(self):
        self.restore()


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

    initial_epoch = 0
    if checkpoint_freq is not None:
        saver_callback = LoadingModelCheckpoint(
            model_dir, period=checkpoint_freq)
        latest_checkpoint = saver_callback.latest_checkpoint
        if latest_checkpoint is not None:
            initial_epoch = saver_callback.filename_epoch(latest_checkpoint)
        callbacks.append(saver_callback)

    if summary_freq:
        kwargs = dict(
            write_graph=False, log_dir=model_dir, update_freq=summary_freq)
        tb_callback = tf.keras.callbacks.TensorBoard(**kwargs)

        # These hacks involve private members - will probably break
        if train_steps_per_epoch is not None and initial_epoch > 0:
            initial_train_steps = \
                initial_epoch*train_steps_per_epoch
            tb_callback._total_batches_seen = initial_train_steps
            # v1 a sample is a batch, where as in v2 a sample is an element
            if is_v1:
                tb_callback._samples_seen = initial_train_steps
            else:
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
