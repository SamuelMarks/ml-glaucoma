from collections import namedtuple

from torchvision.datasets import ImageFolder

from ml_glaucoma import logger
from ml_glaucoma.utils.bmes_data_prep import prepare_bmes_splits

Datasets = namedtuple('Datasets', ('train', 'validation', 'test'))


def get_bmes_builder(data_dir, resolution=(256, 256), rgb=True):
    logger.error('prepare_bmes_splits(data_dir): {};'.format(prepare_bmes_splits(data_dir)))
    return {split: ImageFolder(folder) for (split, folder) in prepare_bmes_splits(data_dir)}
