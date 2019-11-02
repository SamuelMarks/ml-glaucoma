from io import IOBase
from operator import itemgetter
from os import path, listdir

import tensorflow as tf


def log_parser(infile, top, threshold, by_diff, directory, tag='epoch_val_auc'):
    if directory is not None and path.isdir(directory):
        infile = directory
    if not isinstance(infile, IOBase) and path.isdir(infile):
        infile = next(
            filter(lambda fname:
                   path.isfile(fname) and fname.rpartition(path.extsep)[2] not in frozenset(
                       ('h5', 'dot', 'profile-empty')),
                   map(lambda fname: path.join(infile, fname), listdir(infile))))

    sorted_values = sorted(enumerate(v.simple_value
                                     for e in tf.compat.v1.train.summary_iterator(infile)
                                     for v in e.summary.value
                                     if v.tag == tag), key=itemgetter(1), reverse=True)

    print('\n'.join('model-{k:04d}.h5\t{v}'.format(k=k, v=v) for k, v in sorted_values[:top]))

    return sorted_values
