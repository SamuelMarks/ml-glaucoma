from collections import namedtuple
from os import path

from torchvision.datasets import ImageFolder

from ml_glaucoma.constants import IMAGE_RESOLUTION
from ml_glaucoma.utils.bmes_data_prep import prepare_bmes_splits

Datasets = namedtuple("Datasets", ("train", "validation", "test"))


def get_bmes_builder(data_dir, resolution=IMAGE_RESOLUTION, rgb=True):
    # logger.error('prepare_bmes_splits(data_dir): {};'.format(prepare_bmes_splits(data_dir, 'symlinked_datasets')))
    return {
        split: ImageFolder(folder)
        for (split, folder) in prepare_bmes_splits(
            path.join(data_dir, "symlinked_datasets", "bmes")
        )
    }


__all__ = ["get_bmes_builder", "Datasets"]
