from os import path, listdir

import numpy as np
import tensorflow as tf

from ml_glaucoma.constants import SAVE_FORMAT_WITH_SEP
from ml_glaucoma.cli_options.logparser import log_parser


class LoadingModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """
    ModelCheckpoint modified to automatically restore model.

    Weight restoration can be done manually using `self.restore`.

    Restoration only happens once by default, but you can force subsequent
    restorations using `self.restore(force_restore=True)`.
    """

    def __init__(self, model_dir, **kwargs):
        """
        Args:
            model_dir: directory to save weights. Files will have format
                '{model_dir}/{epoch:04d}.h5'.
            **kwargs: passed to `ModelCheckpoint.__init__`.
                All keys valid except `filepath`.
        """
        self._model_dir = model_dir
        self._filename = 'model-{epoch:04d}' + SAVE_FORMAT_WITH_SEP
        super(LoadingModelCheckpoint, self).__init__(
            filepath=path.join(self._model_dir, self._filename), **kwargs)
        self._restored = False

    def restore(self, save_path=None, force_restore=False):
        """
        Restore weights at path, or latest in model directory.

        Does nothing if the model has already been restored by this loader
        unless `force_restore` is True.
        """
        if not self._restored or force_restore:
            if save_path is None:
                save_path = self.latest_checkpoint
            if save_path is not None:
                self.model.load_weights(save_path)
            self._restored = True

    @property
    def latest_checkpoint(self):
        """Get the full path to the latest weights file."""
        filenames = tuple(fn for fn in listdir(self._model_dir)
                          if fn.startswith('model'))
        if len(filenames) == 0:
            return None
        latest = max(filenames, key=LoadingModelCheckpoint.filename_epoch)
        return path.join(self._model_dir, latest)

    @staticmethod
    def filename_epoch(filename):
        """Get the epoch of the given file/path."""
        assert path.splitext(filename) == SAVE_FORMAT_WITH_SEP
        return int(filename[-7:-3])

    def on_train_begin(self, logs=None):
        self._on_begin()

    def on_test_begin(self, logs=None):
        self._on_begin()

    def on_predict_begin(self, logs=None):
        self._on_begin()

    def _on_begin(self):
        self.restore()

    def _save_model(self, epoch, logs):
        # Save for subsequent restoration
        monitor_op, save_best_only, best = self.monitor_op, self.save_best_only, self.best
        # if save_best_only is True: self.save_best_only = False

        filepath = self._get_file_path(epoch, logs)

        def monitor_op(current, _best):
            if self._save_model.t > 0:
                print('current:\t', current,
                      'self.best:\t', _best, sep='')
                print('log_parser:\t', log_parser(path.dirname(filepath), top=epoch, tag='epoch_auc'))
            return np.less

        self.monitor_op = monitor_op

        super(LoadingModelCheckpoint, self)._save_model(epoch, logs)

        #
        # remove(filepath)

        # Restore
        self.monitor_op, self.save_best_only, self.best = monitor_op, save_best_only, best
        pass

    _save_model.t = 3
