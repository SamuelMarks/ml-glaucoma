#!/usr/bin/env python

from os import path

import tensorflow_datasets as tfds

from ml_glaucoma.datasets.tfds_builders.dr_spoc.utils import main


def get_data(dr_spoc_dir):
    dr_spoc_dir, df, filename2cat, combined_df = main(dr_spoc_dir)

    train_ds = tfds.load('mnist:3.*.*', split=tfds.core.ReadInstruction('train'))

    return combined_df


if __name__ == '__main__':
    get_data(dr_spoc_dir=path.join(path.expanduser('~'),
                                   'OneDrive - The University of Sydney (Students)',
                                   'Fundus Photographs for AI',
                                   'DR SPOC Dataset'))
