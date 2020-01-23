from os import path

import numpy as np
import tensorflow as tf

from ml_glaucoma.cli_options.logparser import log_parser


class DropWorseModels(tf.keras.callbacks.ModelCheckpoint):
    """
    Designed around making `save_best_only` work for arbitrary metrics and thresholds between metrics
    """

    def __init__(self, model_dir, monitor, log_dir, **kwargs):
        """
        Args:
            model_dir: directory to save weights. Files will have format
                        '{model_dir}/{epoch:04d}.h5'.
            monitor: quantity to monitor.
            log_dir: the path of the directory where to save the log files to be
                        parsed by TensorBoard.
            **kwargs: passed to `ModelCheckpoint.__init__`.
                        All keys valid except `filepath`.
        """
        self._model_dir = model_dir
        self._filename = 'model-{epoch:04d}.h5'
        self._delete = []
        self._last_epoch_ran = -1
        self._logs = {}
        self._log_dir = log_dir
        super(DropWorseModels, self).__init__(
            filepath=path.join(self._model_dir, self._filename), monitor=monitor, **kwargs
        )

    def _save_model(self, epoch, logs):
        with open('/tmp/log.txt', 'a') as f:
            f.write('\n_save_model::epoch:\t{}\n_save_model::logs:\t{}\n'.format(epoch, logs))

        # Save for subsequent restoration
        monitor_op, save_best_only, best = self.monitor_op, self.save_best_only, self.best
        # if save_best_only is True: self.save_best_only = False

        filepath = self._get_file_path(epoch, logs)

        parsed_log = log_parser(infile=None, top=3, threshold=None, by_diff=None,
                           directory=self._log_dir, rest=None, tag=self.monitor)
        with open('/tmp/log.txt', 'a') as f:
            f.write('_save_model::parsed_log:\t{}\n'.format(parsed_log))
        if epoch in self._logs:
            pass
        else:
            pass
            # if self._logs[epoch]

        # TODO: Run for last epoch

        def monitor_op(current, _best):
            # if self._save_model.t > 0:
            with open('/tmp/log.txt', 'a') as f:
                f.write('current:\t{}\nself.best:\t{}\n'.format(current, _best))
                f.write('log_parser with tag=\'epoch_auc\':\t{}\n\n'.format(log_parser(path.dirname(filepath),
                                                                                       top=epoch, tag='epoch_auc')))
            return np.less

        self.monitor_op = monitor_op
        self._last_epoch_ran = epoch

        super(DropWorseModels, self)._save_model(epoch, logs or parsed_log)

        #
        # remove(filepath)

        # Restore
        self.monitor_op, self.save_best_only, self.best = monitor_op, save_best_only, best

    _save_model.t = 3
