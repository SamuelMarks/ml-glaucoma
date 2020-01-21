from os import path

import numpy as np
import tensorflow as tf

from ml_glaucoma.cli_options.logparser import log_parser


class DropWorseModels(tf.keras.callbacks.ModelCheckpoint):
    """
    Designed around making `save_best_only` work for arbitrary metrics and thresholds between metrics
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
        super(DropWorseModels, self).__init__(
            filepath=path.join(self._model_dir, self._filename), **kwargs)

    def _save_model(self, epoch, logs):
        # Save for subsequent restoration
        monitor_op, save_best_only, best = self.monitor_op, self.save_best_only, self.best
        # if save_best_only is True: self.save_best_only = False

        filepath = self._get_file_path(epoch, logs)

        def monitor_op(current, _best):
            # if self._save_model.t > 0:
            print('current:\t', current,
                  'self.best:\t', _best, sep='')
            print('log_parser with tag=\'epoch_auc\':\t', log_parser(path.dirname(filepath),
                                                                     top=epoch, tag='epoch_auc'))
            return np.less

        self.monitor_op = monitor_op

        super(DropWorseModels, self)._save_model(epoch, logs)

        #
        # remove(filepath)

        # Restore
        self.monitor_op, self.save_best_only, self.best = monitor_op, save_best_only, best

    _save_model.t = 3
