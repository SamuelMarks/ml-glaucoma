from __future__ import division, print_function

import os
import zipfile

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from ml_glaucoma.constants import IMAGE_RESOLUTION
from ml_glaucoma.datasets.tfds_builders import transformer


class RefugeTask(object):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    LOCALIZATION = "localization"

    @classmethod
    def all(cls):
        return (
            RefugeTask.CLASSIFICATION,
            RefugeTask.SEGMENTATION,
            RefugeTask.LOCALIZATION,
        )

    @classmethod
    def validate(cls, task):
        if task not in cls.all():
            raise ValueError(
                "Invalid task '{:s}': must be one of {:s}".format(task, str(cls.all()))
            )


def _load_fovea(archive, subpath):
    import xlrd

    # I struggled to get openpyxl to read already opened zip files
    # necessary to tfds to work with google cloud buckets etc
    # could always decompress the entire directory, but that will result in
    # a much bigger disk space usage
    with archive.open(subpath) as fp:
        wb = xlrd.open_workbook(file_contents=fp.read())
        sheet = wb.sheet_by_index(0)
        data = {}
        for i in range(sheet.ncols):
            col_data = sheet.col(i)
            data[col_data[0].value] = [v.value for v in col_data[1:]]
    return data


def _seg_to_label(seg):
    out = np.zeros(shape=seg.shape[:2] + (1,), dtype=np.uint8)
    out[seg == 128] = 1
    out[seg == 0] = 2
    return out


def _load_image(image_fp):
    return np.array(tfds.core.lazy_imports.PIL_Image.open(image_fp))


def RefugeConfig(resolution, rgb=True):
    return transformer.ImageTransformerConfig(
        description="Refuge grand-challenge dataset", resolution=resolution, rgb=rgb
    )


base_rgb = RefugeConfig(None)
base_gray = RefugeConfig(None, rgb=False)


class Refuge(tfds.core.GeneratorBasedBuilder):
    """
    Glaucoma related dataset builder for REFUGE grand challenge.

    We save data for all tasks in the one set of tfrecords for each resolution.
    This -may- come at a very slight performance penalty and result in slightly
    larger files if one is only concerned with the classification task, but
    makes access more convenience and avoids duplicating data on disk for the
    different tasks.
    """

    BUILDER_CONFIGS = [base_rgb, base_gray]

    URL = "http://refuge.grand-challenge.org"

    def __init__(self, task=RefugeTask.CLASSIFICATION, **kwargs):
        RefugeTask.validate(task)
        self.task = task
        super(Refuge, self).__init__(**kwargs)

    def _info(self):
        resolution = self.builder_config.resolution
        num_channels = 3 if self.builder_config.rgb else 1
        if resolution is None:
            h, w = None, None
        else:
            h, w = resolution
        task = self.task
        label_key = {
            RefugeTask.CLASSIFICATION: "label",
            RefugeTask.SEGMENTATION: "segmentation",
            RefugeTask.LOCALIZATION: "macular_center",
        }[task]
        return tfds.core.DatasetInfo(
            builder=self,
            description=self.builder_config.description,
            features=tfds.features.FeaturesDict(
                {
                    "fundus": tfds.features.Image(shape=(h, w, num_channels)),
                    "segmentation": tfds.features.Image(shape=(h, w, 1)),
                    "label": tfds.features.Tensor(dtype=tf.bool, shape=()),
                    "macular_center": tfds.features.Tensor(
                        dtype=tf.float32, shape=(2,)
                    ),
                    "index": tfds.features.Tensor(dtype=tf.int64, shape=()),
                }
            ),
            homepage=self.URL,
            citation="TODO",
            supervised_keys=("fundus", label_key),
        )

    def _split_generators(self, dl_manager):
        base_url = "https://aipe-broad-dataset.cdn.bcebos.com/gno"
        urls = {
            "train": {
                "fundi": "REFUGE-Training400.zip",
                "annotations": "Annotation-Training400.zip",
            },
            "validation": {
                "fundi": "REFUGE-Validation400.zip",
                "annotations": "REFUGE-Validation400-GT.zip",
            },
            "test": {
                "fundi": "REFUGE-Test400.zip",
            },
        }
        urls = tf.nest.map_structure(  # pylint: disable=no-member
            lambda x: os.path.join(base_url, x), urls
        )
        download_dirs = dl_manager.download(urls)

        return [
            tfds.core.SplitGenerator(
                name=split, gen_kwargs=dict(split=split, **download_dirs[split])
            )
            for split in ("train", "validation", "test")
        ]

    def _generate_examples(self, split, **kwargs):
        return {
            "train": self._generate_train_examples,
            "validation": self._generate_validation_examples,
            "test": self._generate_test_examples,
        }[split](**kwargs)

    def _generate_train_examples(self, fundi, annotations):
        with tf.io.gfile.GFile(annotations, "rb") as annotations:
            annotations = zipfile.ZipFile(annotations)
            fov_data = _load_fovea(
                annotations, "Annotation-Training400/Fovea_location.xlsx"
            )
            xys = {
                fundus_fn: (x, y)
                for fundus_fn, x, y in zip(
                    fov_data["ImgName"], fov_data["Fovea_X"], fov_data["Fovea_Y"]
                )
            }

            with tf.io.gfile.GFile(fundi, "rb") as fundi:
                fundi = zipfile.ZipFile(fundi)

                def get_example(label, _fundus_path, segmentation_path):
                    fundus_fn = _fundus_path.split("/")[-1]
                    xy = np.array(xys[fundus_fn], dtype=np.float32)
                    fundus = _load_image(fundi.open(_fundus_path))
                    seg = _load_image(annotations.open(segmentation_path))
                    seg = _seg_to_label(seg)
                    image_res = fundus.shape[:2]
                    assert seg.shape[:2] == image_res
                    _transformer = self.builder_config.transformer(image_res)
                    if _transformer is not None:
                        xy = _transformer.transform_point(xy)
                        fundus = _transformer.transform_image(
                            fundus, interp=tf.image.ResizeMethod.BILINEAR
                        )
                        seg = _transformer.transform_image(
                            seg, interp=tf.image.ResizeMethod.NEAREST_NEIGHBOR
                        )
                    return {
                        "fundus": fundus,
                        "segmentation": seg,
                        "label": label,
                        "macular_center": xy,
                        "index": index,
                    }

                # positive data_preparation_scripts
                for index in range(1, 41):
                    fundus_path = os.path.join(
                        "Training400", "Glaucoma", "g{:04d}.jpg".format(index)
                    )
                    seg_path = os.path.join(
                        "Annotation-Training400",
                        "Disc_Cup_Masks",
                        "Glaucoma",
                        "g{:04d}.bmp".format(index),
                    )

                    yield (True, index), get_example(True, fundus_path, seg_path)

                # negative data_preparation_scripts
                for index in range(1, 361):
                    fundus_path = os.path.join(
                        "Training400", "Non-Glaucoma", "n{:04d}.jpg".format(index)
                    )
                    seg_path = os.path.join(
                        "Annotation-Training400",
                        "Disc_Cup_Masks",
                        "Non-Glaucoma",
                        "n{:04d}.bmp".format(index),
                    )
                    yield (False, index), get_example(False, fundus_path, seg_path)

    def _generate_validation_examples(self, fundi, annotations):
        with tf.io.gfile.GFile(annotations, "rb") as annotations:
            annotations = zipfile.ZipFile(annotations)
            fov_data = _load_fovea(
                annotations,
                os.path.join("REFUGE-Validation400-GT", "Fovea_locations.xlsx"),
            )
            label_data = {
                fundus_fn: (x, y, bool(label))
                for fundus_fn, x, y, label in zip(
                    fov_data["ImgName"],
                    fov_data["Fovea_X"],
                    fov_data["Fovea_Y"],
                    fov_data["Glaucoma Label"],
                )
            }

            with tf.io.gfile.GFile(fundi, "rb") as fundi:
                fundi = zipfile.ZipFile(fundi)

                for index in range(1, 401):
                    seg_fn = "V{:04d}.bmp".format(index)
                    seg_path = os.path.join(
                        "REFUGE-Validation400-GT", "Disc_Cup_Masks", seg_fn
                    )
                    fundus_fn = "V{:04d}.jpg".format(index)
                    fundus_path = os.path.join("REFUGE-Validation400", fundus_fn)
                    x, y, label = label_data[fundus_fn]
                    xy = np.array([x, y], dtype=np.float32)
                    fundus = _load_image(fundi.open(fundus_path))
                    image_res = fundus.shape[:2]
                    seg = _load_image(annotations.open(seg_path))
                    seg = _seg_to_label(seg)
                    _transformer = self.builder_config.transformer(image_res)
                    if _transformer is not None:
                        xy = _transformer.transform_point(xy)
                        fundus = _transformer.transform_image(
                            fundus, interp=tf.image.ResizeMethod.BILINEAR
                        )
                        seg = _transformer.transform_image(
                            seg, interp=tf.image.ResizeMethod.NEAREST_NEIGHBOR
                        )
                    yield index, {
                        "fundus": fundus,
                        "segmentation": seg,
                        "label": label,
                        "macular_center": xy,
                        "index": index,
                    }

    def _generate_test_examples(self, fundi):
        def get_seg(image_resolution):
            return np.zeros(image_resolution + (1,), dtype=np.uint8)

        xy = -np.ones((2,), dtype=np.float32)

        with tf.io.gfile.GFile(fundi, "rb") as fundi:
            fundi = zipfile.ZipFile(fundi)
            for index in range(1, 401):
                fundus = _load_image(
                    fundi.open(os.path.join("Test400", "T{:04d}.jpg".format(index)))
                )
                image_res = fundus.shape[:2]
                _transformer = self.builder_config.transformer(image_res)
                if _transformer is not None:
                    fundus = _transformer.transform_image(
                        fundus, interp=tf.image.ResizeMethod.BILINEAR
                    )
                yield index, {
                    "fundus": fundus,
                    "segmentation": get_seg(fundus.shape[:2]),
                    "label": False,
                    "macular_center": xy,
                    "index": index,
                }


def get_refuge_builder(resolution=IMAGE_RESOLUTION, rgb=True, data_dir=None):
    if resolution is None:
        config = base_rgb if rgb else base_gray
    else:
        config = RefugeConfig(resolution, rgb)
    return Refuge(config=config, data_dir=data_dir)


__all__ = ["Refuge", "RefugeTask", "get_refuge_builder"]
