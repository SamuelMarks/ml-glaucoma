import functools

from ml_glaucoma import problems as p
from ml_glaucoma.cli_options.base import Configurable


class ConfigurableMapFn(Configurable):
    def fill_self(self, parser):
        parser.add_argument(
            '-fv', '--maybe_vertical_flip', action='store_true',
            help='randomly flip training input vertically')
        parser.add_argument(
            '-fh', '--maybe_horizontal_flip', action='store_true',
            help='randomly flip training input horizontally')
        parser.add_argument(
            '--gray', help='use grayscale', action='store_true')

    def build_self(self, gray, maybe_horizontal_flip, maybe_vertical_flip, **kwargs):
        val_map_fn = functools.partial(
            p.preprocess_example,
            use_rgb=not gray,
            per_image_standardization=True)

        train_map_fn = functools.partial(
            val_map_fn,
            maybe_horizontal_flip=maybe_horizontal_flip,
            maybe_vertical_flip=maybe_vertical_flip)
        map_fn = {
            'train': train_map_fn,
            'validation': val_map_fn,
            'test': val_map_fn
        }
        return map_fn


__all__ = ['ConfigurableMapFn']
