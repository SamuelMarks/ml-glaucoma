from io import IOBase
from operator import itemgetter
from os import path, listdir

import tensorflow as tf


def log_parser(infile, top, threshold, by_diff, directory, tag='epoch_val_auc'):
    if directory is not None and path.isdir(directory):
        infile = directory

    files = []

    def process_dir(_directory):
        for fname in listdir(_directory):
            full_path = path.join(_directory, fname)
            if path.isfile(full_path) and full_path.rpartition(path.extsep
                                                               )[2] not in frozenset(('h5', 'dot',
                                                                                      'profile-empty', 'trace')):
                files.append(full_path)
            elif path.isdir(full_path):
                process_dir(full_path)

    if isinstance(infile, IOBase):
        files.append(infile)
    elif path.isdir(directory):
        process_dir(directory)

    for fname in files:
        sorted_values = sorted(enumerate(v.simple_value
                                         for e in tf.compat.v1.train.summary_iterator(fname)
                                         for v in e.summary.value
                                         if v.tag == tag), key=itemgetter(1), reverse=True)

        if len(sorted_values):
            print('\n'.join('model-{k:04d}.h5\t{v}'.format(k=k, v=v) for k, v in sorted_values[:top]))
