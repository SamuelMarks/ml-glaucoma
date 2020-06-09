from __future__ import division
from __future__ import print_function

from ml_glaucoma.datasets.tfds_builders.base import base_builder
from ml_glaucoma.utils.bmes_data_prep import get_data


def bmes_builder(data_dir, init,
                 parent_dir, manual_dir,
                 force_create=False):
    return base_builder(dataset_name='bmes',
                        data_dir=data_dir,
                        init=init,
                        parent_dir=parent_dir,
                        manual_dir=manual_dir,
                        get_data=get_data,
                        force_create=force_create,
                        supported_names=frozenset(('bmes',)))


__all__ = ['base_builder']
