from io import IOBase
from os import path, listdir

import tensorflow as tf

from ml_glaucoma.utils import sorted_enumerate


def log_parser(infile, top, threshold, by_diff, tag='epoch_val_auc'):
    if not isinstance(infile, IOBase) and path.isdir(infile):
        infile = next(fname for fname in listdir(infile)
                      if path.isfile(fname) and fname.rpartition(path.extsep)[2] not in frozenset(('h5', 'dot')))

    i = 0
    for e in tf.train.summary_iterator(infile):
        for v in e.summary.value:
            if v.tag == tag:
                i += 1
                print('model-{:04d}.h5'.format(i), v.simple_value, sep='\t')

    sorted_vals = sorted_enumerate(v.simple_value
                                   for e in tf.train.summary_iterator(infile)
                                   for v in e.summary.value
                                   if v.tag == tag)
    print(sorted_vals)
