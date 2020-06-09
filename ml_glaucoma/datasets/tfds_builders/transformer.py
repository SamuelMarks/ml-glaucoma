from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds


class Transformer(object):
    def __init__(self, initial_res, target_res, rgb):
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
        self.rgb = rgb

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
        if not self.rgb and image.shape[-1] == 3:
            image = np.mean(image, axis=-1, keepdims=True)
        return image

    def transform_point(self, xy):
        xy = xy.copy()
        if self.delta is not None:
            xy += self.delta
        if self.scale != 1:
            xy *= self.scale
        return xy


class ImageTransformerConfig(tfds.core.BuilderConfig):
    def __init__(
        self, description, name=None, resolution=None, rgb=True,
        version=tfds.core.Version("0.0.1")):
        color_suffix = 'rgb' if rgb else 'gray'
        if resolution is None:
            self.resolution = None
            if name is None:
                name = 'raw-{:s}'.format(color_suffix)
            desc_suffix = ' ({:s})'.format(color_suffix)
        else:
            if isinstance(resolution, int):
                resolution = (resolution,) * 2
            else:
                resolution = tuple(resolution)
            if not all(isinstance(r, int) for r in resolution):
                raise ValueError("`resolution`s must be `None` or all `int`s, got {!r}".format(resolution))
            self.resolution = resolution
            if name is None:
                name = 'r{:d}-{:d}-{:s}'.format(resolution[0], resolution[1], color_suffix)
            desc_suffix = " ({:d} x {:d}, {:s})".format(resolution[0], resolution[1], color_suffix)
        self.rgb = rgb

        super(ImageTransformerConfig, self).__init__(
            name=name,
            version=version,
            description="{:s}{:s}".format(description, desc_suffix)
        )

    def transformer(self, image_resolution):
        if self.resolution is None or image_resolution == self.resolution:
            return None
        else:
            return Transformer(image_resolution, self.resolution, self.rgb)


__all__ = ['Transformer', 'ImageTransformerConfig']
