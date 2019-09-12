import os

import tensorflow as tf


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
        self._filename = 'model-{epoch:04d}.h5'
        super(LoadingModelCheckpoint, self).__init__(
            filepath=os.path.join(self._model_dir, self._filename), **kwargs)
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
        filenames = tuple(fn for fn in os.listdir(self._model_dir)
                          if fn.startswith('model'))
        if len(filenames) == 0:
            return None
        latest = max(filenames, key=LoadingModelCheckpoint.filename_epoch)
        return os.path.join(self._model_dir, latest)

    @staticmethod
    def filename_epoch(filename):
        """Get the epoch of the given file/path."""
        assert (filename.endswith('.h5'))
        return int(filename[-7:-3])

    def on_train_begin(self, logs=None):
        self._on_begin()

    def on_test_begin(self, logs=None):
        self._on_begin()

    def on_predict_begin(self, logs=None):
        self._on_begin()

    def _on_begin(self):
        self.restore()
