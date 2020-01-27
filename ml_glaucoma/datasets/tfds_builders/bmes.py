from __future__ import division
from __future__ import print_function

import os
from os import path
from platform import python_version_tuple

from ml_glaucoma.constants import IMAGE_RESOLUTION
from ml_glaucoma.utils.bmes_data_prep import prepare_bmes_splits

if python_version_tuple()[0] == '2':
    from itertools import ifilter as filter

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from ml_glaucoma.datasets.tfds_builders import transformer


def BmesConfig(resolution=None, rgb=True):
    return transformer.ImageTransformerConfig(
        description="TODO", resolution=resolution, rgb=rgb)


def _load_image(image_fp):
    return np.array(tfds.core.lazy_imports.PIL_Image.open(image_fp))


base_rgb = BmesConfig(rgb=True)
base_gray = BmesConfig(rgb=False)


class Bmes(tfds.core.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [base_rgb, base_gray]

    def _info(self):
        resolution = self.builder_config.resolution
        if resolution is None:
            h, w = None, None
        else:
            h, w = resolution
        return tfds.core.DatasetInfo(
            builder=self,
            description=self.builder_config.description,
            features=tfds.features.FeaturesDict({
                "fundus": tfds.features.Image(shape=(h, w, 3)),
                "label": tfds.features.Tensor(dtype=tf.bool, shape=()),
                "filename": tfds.features.Text(),
            }),
            homepage="https://github.com/SamuelMarks/ml-glaucoma",
            citation="TODO",
            supervised_keys=("fundus", "label")
        )

    def _make_download_manager(self, download_dir, download_config):
        """
        We override the base _make_download_manager to adjust manual_dir.

        The default behaviour is to adjust the input download_config's
        manual_dir by appending the builder's name. We remove appending.

        Issue raised at https://github.com/tensorflow/datasets/issues/587
        """
        from tensorflow_datasets.core import download
        download_dir = download_dir or os.path.join(
            self._data_dir_root, "downloads")
        extract_dir = (download_config.extract_dir or
                       os.path.join(download_dir, "extracted"))
        manual_dir = (download_config.manual_dir or
                      os.path.join(download_dir, "manual"))

        # if test below is the only difference from original
        if download_config.manual_dir is None:
            manual_dir = os.path.join(manual_dir, self.name)

        force_download = (
            download_config.download_mode == download.GenerateMode.FORCE_REDOWNLOAD
        )
        return download.DownloadManager(
            dataset_name=self.name,
            download_dir=download_dir,
            extract_dir=extract_dir,
            manual_dir=manual_dir,
            force_download=force_download,
            force_extraction=force_download,
            register_checksums=download_config.register_checksums,
        )

    def _split_generators(self, dl_manager):
        manual_dir = dl_manager.manual_dir

        return list(filter(
            lambda split_folder: tfds.core.SplitGenerator(
                name=split_folder.split,
                gen_kwargs=dict(folder=split_folder.folder)
            ),
            prepare_bmes_splits(path.join(manual_dir, 'symlinked_datasets', 'bmes'))
        ))

    def _generate_examples(self, folder):
        i = -1
        for dirname, label in (('no_glaucoma', False), ('glaucoma', True)):
            subdir = os.path.join(folder, dirname)
            for filename in os.listdir(subdir):
                if not filename.endswith('.jpg'):
                    raise IOError('All files in directory must be `.jpg`')
                path = os.path.join(subdir, filename)
                with tf.io.gfile.GFile(path, "rb") as fp:
                    fundus = _load_image(fp)
                curr_transformer = self.builder_config.transformer(fundus.shape[:2])
                if curr_transformer is not None:
                    fundus = curr_transformer.transform_image(
                        fundus, interp=tf.image.ResizeMethod.BILINEAR)
                i += 1
                yield i, dict(
                    fundus=fundus,
                    label=label,
                    filename=filename,
                )


def get_bmes_builder(resolution=IMAGE_RESOLUTION, rgb=True, data_dir=None):
    if resolution is None:
        config = base_rgb if rgb else base_gray
    else:
        config = BmesConfig(resolution, rgb)
    return Bmes(config=config, data_dir=data_dir)
