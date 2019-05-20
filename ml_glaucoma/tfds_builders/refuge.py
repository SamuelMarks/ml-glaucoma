from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets.public_api as tfds
import tensorflow as tf
import numpy as np
import os
import zipfile


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
                "Invalid task '%s': must be one of %s"
                % (task, str(cls.all())))


class Transformer(object):
    def __init__(self, initial_res, target_res):
        th, tw = target_res
        ih, iw = initial_res
        tr = tw / th
        ir = iw / ih
        if ir < tr:
            pad_w = int(tw * ih / th) - iw
            pad_left = pad_w // 2
            delta = np.array([pad_left, 0], dtype=np.float32)
            padding = [
                [0, 0],
                [pad_left, pad_w - pad_left],
                [0, 0]
            ]
            iw += pad_w
        elif ir > tr:
            pad_h = int(th * iw / tw) - ih
            pad_top = pad_h // 2
            padding = [
                [pad_top, pad_h - pad_top],
                [0, 0],
                [0, 0]
            ]
            delta = np.array([0, pad_top], dtype=np.float32)
            ih += pad_h
        else:
            padding = None
            delta = None

        self.scale = th / ih
        self.padding = padding
        self.delta = delta
        self.target_res = target_res

    def transform_image(self, image, interp=tf.image.ResizeMethod.BILINEAR):
        Image = tfds.core.lazy_imports.PIL_Image
        if self.padding is not None:
            image = np.pad(image, self.padding, mode='constant')
        if self.scale != 1:
            resample = {
                tf.image.ResizeMethod.NEAREST_NEIGHBOR: Image.NEAREST,
                tf.image.ResizeMethod.BILINEAR: Image.BILINEAR,
            }[interp]

            if image.shape[-1] == 1:
                image = np.squeeze(image, axis=-1)
            image = Image.fromarray(image)
            image = np.array(image.resize(self.target_res, resample=resample))
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
        return image

    def transform_point(self, xy):
        xy = xy.copy()
        if self.delta is not None:
            xy += self.delta
        if self.scale != 1:
            xy *= self.scale
        return xy



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
    out[seg == 128] == 1
    out[seg == 0] = 2
    return out


def _load_image(image_fp):
    return np.array(tfds.core.lazy_imports.PIL_Image.open(image_fp))


class RefugeConfig(tfds.core.BuilderConfig):
    def __init__(self, resolution=None, version=tfds.core.Version("0.0.1")):
        if resolution is None:
            self.resolution = None
            name = 'raw'
            desc_suffix = ''
        else:
            resolution = tuple(resolution)
            if not all(isinstance(r, int) for r in resolution):
                raise ValueError("All resolutions must be ints")
            self.resolution = resolution
            name = 'r%d-%d' % resolution
            desc_suffix = " (%d x %d)" % resolution

        super(RefugeConfig, self).__init__(
            name=name,
            version=version,
            description="Refuge grand-challenge dataset%s" % desc_suffix)


    def transformer(self, image_resolution):
        if self.resolution is None or image_resolution == self.resolution:
            return None
        else:
            return Transformer(image_resolution, self.resolution)


class Refuge(tfds.core.GeneratorBasedBuilder):
    """
    Glaucoma related dataset builder for REFUGE grand challenge.

    We save data for all tasks in the one set of tfrecords for each resolution.
    This -may- come at a very slight performance penalty and result in slightly
    larger files if one is only concerned with the classification task, but
    makes access more convenience and avoids duplicating data on disk for the
    different tasks.
    """

    BUILDER_CONFIGS = [RefugeConfig(), RefugeConfig((256, 256))]

    URL = "http://refuge.grand-challenge.org"

    def __init__(self, task=RefugeTask.CLASSIFICATION, **kwargs):
        RefugeTask.validate(task)
        self.task = task
        super(Refuge, self).__init__(**kwargs)

    def _info(self):
        resolution = self.builder_config.resolution
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
            features=tfds.features.FeaturesDict({
                "fundus": tfds.features.Image(shape=(h, w, 3)),
                "segmentation": tfds.features.Image(shape=(h, w, 1)),
                "label": tfds.features.Tensor(dtype=tf.bool, shape=()),
                "macular_center": tfds.features.Tensor(
                    dtype=tf.float32, shape=(2,)),
                "index": tfds.features.Tensor(dtype=tf.int64, shape=()),
            }),
            urls=[self.URL],
            citation="TODO",
            supervised_keys=("fundus", label_key)
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
            }
        }
        urls = tf.nest.map_structure( # pylint: disable=no-member
            lambda x: os.path.join(base_url, x), urls)
        download_dirs = dl_manager.download(urls)

        return [
            tfds.core.SplitGenerator(
              name=split,
              num_shards=4,
              gen_kwargs=dict(split=split, **download_dirs[split]),
          ) for split in ("train", "validation", "test")
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
                annotations, "Annotation-Training400/Fovea_location.xlsx")
            xys = {
                fundus_fn: (x, y) for fundus_fn, x, y in zip(
                    fov_data["ImgName"],
                    fov_data["Fovea_X"],
                    fov_data["Fovea_Y"])}

            with tf.io.gfile.GFile(fundi, "rb") as fundi:
                fundi = zipfile.ZipFile(fundi)

                def get_example(label, fundus_path, segmentation_path):
                    fundus_fn = fundus_path.split("/")[-1]
                    xy = np.array(xys[fundus_fn], dtype=np.float32)
                    fundus = _load_image(fundi.open(fundus_path))
                    seg = _load_image(annotations.open(segmentation_path))
                    seg = _seg_to_label(seg)
                    image_res = fundus.shape[:2]
                    assert(seg.shape[:2] == image_res)
                    transformer = self.builder_config.transformer(image_res)
                    if transformer is not None:
                        xy = transformer.transform_point(xy)
                        fundus = transformer.transform_image(
                            fundus, interp=tf.image.ResizeMethod.BILINEAR)
                        seg = transformer.transform_image(
                            seg, interp=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    return {
                        "fundus": fundus,
                        "segmentation": seg,
                        "label": label,
                        "macular_center": xy,
                        "index": index,
                    }

                # positive examples
                for index in range(1, 41):
                    fundus_path = os.path.join(
                        "Training400", "Glaucoma",
                        "g%04d.jpg" % index)
                    seg_path = os.path.join(
                         "Annotation-Training400", "Disc_Cup_Masks",
                          "Glaucoma", "g%04d.bmp" % index)

                    yield get_example(True, fundus_path, seg_path)

                # negative examples
                for index in range(1, 361):
                    fundus_path = os.path.join(
                        "Training400", "Non-Glaucoma",
                        "n%04d.jpg" % index)
                    seg_path = os.path.join(
                         "Annotation-Training400", "Disc_Cup_Masks",
                          "Non-Glaucoma", "n%04d.bmp" % index)
                    yield get_example(False, fundus_path, seg_path)

    def _generate_validation_examples(self, fundi, annotations):
        with tf.io.gfile.GFile(annotations, "rb") as annotations:
            annotations = zipfile.ZipFile(annotations)
            fov_data = _load_fovea(
                annotations, "REFUGE-Validation400-GT/Fovea_locations.xlsx")
            label_data = {
                fundus_fn: (x, y, bool(label)) for fundus_fn, x, y, label in
                    zip(
                        fov_data["ImgName"],
                        fov_data["Fovea_X"],
                        fov_data["Fovea_Y"],
                        fov_data["Glaucoma Label"])
            }

            with tf.io.gfile.GFile(fundi, "rb") as fundi:
                fundi = zipfile.ZipFile(fundi)

                for index in range(1, 401):
                    seg_fn = "V%04d.bmp" % index
                    seg_path = os.path.join(
                        "REFUGE-Validation400-GT", "Disc_Cup_Masks", seg_fn)
                    fundus_fn = "V%04d.jpg" % index
                    fundus_path = os.path.join(
                        "REFUGE-Validation400", fundus_fn)
                    x, y, label = label_data[fundus_fn]
                    xy = np.array([x, y], dtype=np.float32)
                    fundus = _load_image(fundi.open(fundus_path))
                    image_res = fundus.shape[:2]
                    seg = _load_image(annotations.open(seg_path))
                    seg = _seg_to_label(seg)
                    transformer = self.builder_config.transformer(image_res)
                    if transformer is not None:
                        xy = transformer.transform_point(xy)
                        fundus = transformer.transform_image(
                            fundus, interp=tf.image.ResizeMethod.BILINEAR)
                        seg = transformer.transform_image(
                            seg, interp=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    yield {
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
                fundus = _load_image(fundi.open("Test400/T%04d.jpg" % index))
                image_res = fundus.shape[:2]
                transformer = self.builder_config.transformer(image_res)
                if transformer is not None:
                    fundus = transformer.transform_image(
                        fundus, interp=tf.image.ResizeMethod.BILINEAR)
                yield {
                    "fundus": fundus,
                    "segmentation": get_seg(fundus.shape[:2]),
                    "label": False,
                    "macular_center": xy,
                    "index": index,
                }
