import os
from os import path
from tempfile import mkdtemp

import tensorflow as tf
import tensorflow_datasets as tfds

from ml_glaucoma import get_logger

logger = get_logger('.'.join((path.basename(path.dirname(__file__)),
                              path.basename(__file__).rpartition('.')[0])))


def base_builder(dataset_name, data_dir, init,
                 parent_dir, manual_dir,
                 get_data,
                 force_create=False,
                 supported_names=frozenset()):  # type: (str,str,str,str,str,((str, str) -> None),bool, frozenset) -> (((int, bool, str) -> (tfds.image.ImageLabelFolder)), str, str)
    assert dataset_name in supported_names, '{!r} not found in {!r}'.format(dataset_name, supported_names)

    if init:
        if manual_dir is None:
            raise ValueError(
                '`manual_dir` must be provided if `init is True`')
        elif parent_dir is None:
            raise ValueError(
                '`parent_dir` must be provided if '
                '`init is True`')
        elif force_create or not path.isdir(path.join(_get_manual_dir(parent_dir, manual_dir), dataset_name)):
            get_data(parent_dir, manual_dir)
        else:
            logger.info('Using already created symlinks')

        part = 'tensorflow_datasets'
        if not data_dir.endswith(part):
            data_dir = path.join(data_dir, part)

        manual_dir = _get_manual_dir(parent_dir, manual_dir)
        assert path.isdir(manual_dir), 'Manual directory {!r} does not exist. ' \
                                       'Create it and download/extract dataset artifacts ' \
                                       'in there. Additional instructions: ' \
                                       'This is a \'template\' dataset.'.format(
            manual_dir
        )

    def builder_factory(resolution, rgb, data_dir):  # type: (int, bool, str) -> tfds.image.ImageLabelFolder
        print('resolution:'.ljust(20), '{!r}'.format(resolution), sep='')

        class BaseImageLabelFolder(tfds.image.ImageLabelFolder):
            def _info(self):
                return tfds.core.DatasetInfo(
                    builder=self,
                    description='TODO',
                    features=tfds.features.FeaturesDict({
                        'image': tfds.features.Image(  # shape=resolution + ((3 if rgb else 1),),
                            encoding_format='jpeg'),
                        'label': tfds.features.ClassLabel(num_classes=3 if dataset_name == 'dr_spoc' else 2)
                    }),
                    supervised_keys=('image', 'label'),
                )

            def _generate_examples(self, label_images):
                """Generate example for each image in the dict."""

                temp_dir = mkdtemp(prefix='dr_spoc')
                for label, image_paths in label_images.items():
                    for image_path in image_paths:
                        key = '/'.join((label, os.path.basename(image_path)))

                        temp_image_filename = path.join(temp_dir, key.replace(path.sep, '_'))

                        if base_builder.session._closed:
                            base_builder.session = tf.compat.v1.Session()
                            base_builder.session.__enter__()

                        image_decoded = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3 if rgb else 1)
                        resized = tf.image.resize(image_decoded, resolution)
                        enc = tf.image.encode_jpeg(tf.cast(resized, tf.uint8),
                                                   'rgb' if rgb else 'grayscale',
                                                   quality=100, chroma_downsampling=False)
                        fwrite = tf.io.write_file(tf.constant(temp_image_filename), enc)
                        result = base_builder.session.run(fwrite)

                        yield key, {
                            'image': temp_image_filename,
                            'label': label,
                        }

                print('resolved all files, now you should delete: {!r}'.format(temp_dir))
                if not base_builder.session._closed:
                    base_builder.session.__exit__(None, None, None)

        builder = BaseImageLabelFolder(
            dataset_name=dataset_name,
            data_dir=data_dir
        )

        return builder

    return builder_factory, data_dir, manual_dir


base_builder.session = type('FakeSession', tuple(), {'_closed': True})()


def _get_manual_dir(parent_dir, manual_dir):  # type: (str, str) -> str
    if path.dirname(manual_dir) != 'DR SPOC Dataset' \
        and not path.isdir(path.join(manual_dir, 'DR SPOC')) \
        and not path.isdir(path.join(path.dirname(manual_dir), 'DR SPOC')):
        symlinked_datasets_directory = path.join(parent_dir,
                                                 'symlinked_datasets')
        manual_dir = symlinked_datasets_directory
    return manual_dir
