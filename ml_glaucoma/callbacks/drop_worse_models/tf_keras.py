from operator import itemgetter
from os import path, listdir, remove

import tensorflow as tf

from ml_glaucoma.cli_options.logparser import log_parser
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
        self._filename = 'model-{:04d}.h5'
        self._log_dir = log_dir
        self._keep_best = keep_best
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        super(DropWorseModels, self).on_epoch_end(epoch, logs)
        if epoch < self._keep_best: return

        tf_events_logs = log_parser(infile=None, top=epoch, threshold=None, by_diff=None,
                                    directory=self._log_dir, rest=None, tag=self.monitor,
                                    output=False)[1][:self._keep_best]
        keep_models = frozenset(map(self._filename.format, map(itemgetter(0), tf_events_logs)))

        remove_these = tuple(map(lambda filename: path.join(self._model_dir, filename),
                                 filter(lambda filename: filename not in keep_models,
                                        filter(lambda filename: filename.endswith('{}h5'.format(path.extsep)),
                                               listdir(self._model_dir)))))
        out_of = listdir(self._model_dir)

        h5_files = frozenset(filter(lambda filename: filename.endswith('{}h5'.format(path.extsep)),
                                    listdir(self._model_dir)))

        #it_consumes(map(lambda filename: remove(path.join(self._model_dir, filename)),
        #                frozenset(filter(lambda filename: filename.endswith('{}h5'.format(path.extsep)),
        #                                 listdir(self._model_dir))
        #                          ) - keep_models))

        # it_consumes(map(remove,
        #                map(lambda filename: path.join(self._model_dir, filename),
        #                    filter(lambda filename: filename not in keep_models,
        #                           h5_files))))
        with open('/tmp/log.txt', 'a') as f:
            f.write('\n\n_save_model::epoch:\t{}'
                    '\n_save_model::logs:\t{}'
                    '\n_save_model::tf_events_logs:\t{}'
                    '\n_save_model::keep_models:\t{}'
                    '\n_save_model::remove_these:\t{}'
                    '\n_save_model::out_of:\t{}'
                    '\n_save_model::h5_files - keep_models:\t{}'
                    '\n_save_model::keep_models - h5_files:\t{}'
                    '\n'.format(epoch,
                                logs,
                                tf_events_logs,
                                keep_models,
                                remove_these,
                                out_of,
                                h5_files - keep_models,
                                keep_models - h5_files))
