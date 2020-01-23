from io import IOBase
from operator import itemgetter
from os import path, listdir
from sys import stderr

import tensorflow as tf
from tensorflow.core.util import event_pb2

from ml_glaucoma.constants import SAVE_FORMAT_WITH_SEP


def log_parser(infile, top, directory, stdout=True,
               threshold=None, by_diff=None, rest=None,
               tag='epoch_val_auc'):  # type: (IOBase, int, int, int, str, [str], str) -> (str, [(float, float)])
    if directory is not None and path.isdir(directory):
        infile = directory  # type: str

    files = []  # type: [str]

    def process_dir(_directory):
        for fname in listdir(_directory):
            full_path = path.join(_directory, fname)  # type: str
            if path.isfile(full_path) and full_path.rpartition(path.extsep
                                                               )[2] not in frozenset(('h5', 'tf', 'dot',
                                                                                      'profile-empty', 'trace')):
                files.append(full_path)
            elif path.isdir(full_path):
                process_dir(full_path)

    if isinstance(infile, IOBase):
        files.append(infile)
    elif path.isdir(directory):
        process_dir(directory)

    for fname in files:
        total_images = 0
        try:
            total_images += sum(1 for _ in tf.data.TFRecordDataset(fname))  # Check corrupted tf records
        except:
            print("{} in {} is corrupted".format(fname, directory), file=stderr)
        else:
            pass
            # print("{} in {} is not corrupted".format(fname, directory), file=stderr)
        # print("Succeed, tf records found for {} images".format(total_images), file=stderr)

        dirn = path.dirname(directory).rpartition(path.sep)[2]

        if tag == 'all':
            last_result = None

            for record in tf.data.TFRecordDataset(fname):
                event = event_pb2.Event.FromString(tf.get_static_value(record))
                if event.HasField('summary'):
                    value = event.summary.value.pop(0)
                    last_result = {
                        'idx': event.step,
                        'simple_value': value.simple_value,
                        'tag': value.tag,
                        'dirn': dirn
                    }
                    stdout and print('{idx:04d}\t{simple_value:09f}\t{tag:>20}\t{dirn}'.format(**last_result))

            return last_result, (last_result,)
        else:
            values = []
            for record in tf.data.TFRecordDataset(fname):
                event = event_pb2.Event.FromString(tf.get_static_value(record))
                if event.HasField('summary'):
                    value = event.summary.value.pop(0)
                    if value.tag == tag:
                        values.append(value.simple_value)

            sorted_values = tuple(sorted(enumerate(values), key=itemgetter(1), reverse=True))

            stdout and print('\n'.join('{dirn}\tmodel-{k:04d}{ext}\t{v}'.format(dirn=dirn.ljust(54),
                                                                                k=k, v=v, ext=SAVE_FORMAT_WITH_SEP)
                                       for k, v in sorted_values[:top]))
            last_result = fname, sorted_values[:top]

            return last_result, sorted_values
