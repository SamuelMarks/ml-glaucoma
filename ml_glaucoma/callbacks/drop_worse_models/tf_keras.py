from itertools import islice
from operator import itemgetter
from os import path, listdir, remove

import tensorflow as tf

from ml_glaucoma.cli_options.logparser import log_parser
from ml_glaucoma.constants import SAVE_FORMAT_WITH_SEP
from ml_glaucoma.utils import it_consumes


class DropWorseModels(tf.keras.callbacks.Callback):
    """
    Designed around making `save_best_only` work for arbitrary metrics and thresholds between metrics
    """

    def __init__(self, model_dir, monitor, log_dir, keep_best=2):
        """
        Args:
            model_dir: directory to save weights. Files will have format
                        '{model_dir}/{epoch:04d}.h5'.
            monitor: quantity to monitor.
            log_dir: the path of the directory where to save the log files to be
                        parsed by TensorBoard.
            keep_best: number of models to keep, sorted by monitor value
        """
        super(DropWorseModels, self).__init__()
        self._model_dir = model_dir
        self._filename = 'model-{:04d}' + SAVE_FORMAT_WITH_SEP
        self._log_dir = log_dir
        self._keep_best = keep_best
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        super(DropWorseModels, self).on_epoch_end(epoch, logs)
        if epoch < self._keep_best:
            return

        model_files = frozenset(filter(lambda filename: path.splitext(filename)[1] == SAVE_FORMAT_WITH_SEP,
                                       listdir(self._model_dir)))

        if len(model_files) < self._keep_best:
            return

        tf_events_logs = tuple(islice(log_parser(infile=None,
                                                 top=min(self._keep_best, epoch),
                                                 directory=self._log_dir,
                                                 tag=self.monitor,
                                                 stdout=False)[1],
                                      # 0,
                                      self._keep_best))
        keep_models = frozenset(map(self._filename.format, map(itemgetter(0), tf_events_logs)))

        if len(keep_models) < self._keep_best:
            return

        files_to_remove = model_files - keep_models

        with open('/tmp/log.txt', 'a') as f:
            f.write('\n\n_save_model::epoch:                  \t{}'.format(epoch).ljust(30))
            f.write('\n_save_model::logs:                     \t{}'.format(logs).ljust(30))
            f.write('\n_save_model::tf_events_logs:           \t{}'.format(tf_events_logs).ljust(30))
            f.write('\n_save_model::keep_models:              \t{}'.format(keep_models).ljust(30))
            f.write('\n_save_model::model_files:              \t{}'.format(model_files).ljust(30))
            f.write('\n_save_model::model_files - keep_models:\t{}'.format(files_to_remove).ljust(30))
            f.write('\n_save_model::keep_models - model_files:\t{}\n'.format(keep_models - model_files).ljust(30))

        it_consumes(
            islice(map(lambda filename: remove(path.join(self._model_dir, filename)),
                       files_to_remove),
                   # 0,
                   len(keep_models) - self._keep_best
                   )
        )
