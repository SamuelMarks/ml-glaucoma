import os
from os import path

import tensorflow as tf
import tensorflow_datasets as tfds

from ml_glaucoma import get_logger

logger = get_logger('.'.join((path.basename(path.dirname(__file__)),
                              path.basename(__file__).rpartition('.')[0])))

dr_spoc_datasets = 'dr_spoc', 'dr_spoc_grad_and_no_grad', 'dr_spoc_no_no_grad_dir'
dr_spoc_datasets_set = frozenset(dr_spoc_datasets)


def dr_spoc_builder(dataset_name, data_dir, dr_spoc_init,
                    dr_spoc_parent_dir, manual_dir,
                    force_create=False):  # type: (str,str,str,str,str,bool) -> (((int, bool, str) -> (tfds.image.ImageLabelFolder)), str, str)
    assert dataset_name in dr_spoc_datasets_set, '{!r} not found in {!r}'.format(dataset_name, dr_spoc_datasets_set)

    if dr_spoc_init:
        from ml_glaucoma.utils.dr_spoc_data_prep import get_data

        if manual_dir is None:
            raise ValueError(
                '`manual_dir` must be provided if `dr_spoc_init is True`')
        elif dr_spoc_parent_dir is None:
            raise ValueError(
                '`dr_spoc_parent_dir` must be provided if '
                '`dr_spoc_init is True`')
        elif force_create or not path.isdir(path.join(_get_manual_dir(dr_spoc_parent_dir, manual_dir), dataset_name)):
            get_data(root_directory=dr_spoc_parent_dir, manual_dir=manual_dir)
        else:
            logger.info('Using already created symlinks')

        part = 'tensorflow_datasets'
        if not data_dir.endswith(part):
            data_dir = path.join(data_dir, part)

        just = 20
        if dr_spoc_builder.t > 0:
            dr_spoc_builder.t -= 1
            print(
                'data_dir:'.ljust(just), '{!r}\n'.format(data_dir),
                'manual_dir:'.ljust(just), '{!r}\n'.format(manual_dir),
                '_get_manual_dir:'.ljust(just), '{!r}\n'.format(_get_manual_dir(dr_spoc_parent_dir, manual_dir)),
                sep=''
            )

        manual_dir = _get_manual_dir(dr_spoc_parent_dir, manual_dir)
        assert path.isdir(manual_dir), 'Manual directory {!r} does not exist. ' \
                                       'Create it and download/extract dataset artifacts ' \
                                       'in there. Additional instructions: ' \
                                       'This is a \'template\' dataset.'.format(
            manual_dir
        )

    # DrSpocImageLabelFolder.BUILDER_CONFIGS.append(
    #
    # )

    # manual_dir = path.join(bmes_parent_dir, 'tensorflow_datasets')
    # print(builder.info)  # Splits, num examples,... automatically extracted
    # ds = builder.as_dataset(split=('test', 'train', 'valid'), shuffle_files=True)
    # builders.append(builder)
    #
    # return
    # print('ml_glaucoma/cli_options/prepare/tf_keras.py::data_dir: {!r}'.format(data_dir))
    # TODO: Ensure resolution, RGB can be provided
    def builder_factory(resolution, rgb, data_dir):  # type: (int, bool, str) -> tfds.image.ImageLabelFolder
        # builder._data_dir = data_dir
        print('resolution:'.ljust(20), '{!r}'.format(resolution), sep='')

        class DrSpocImageLabelFolder(tfds.image.ImageLabelFolder):
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

                from tempfile import mkdtemp

                # tempdir = mkdtemp(prefix='dr_spoc')  # TODO: Cleanup

                def decode_img(img):
                    # convert the compressed string to a 3D uint8 tensor
                    img = tf.image.decode_jpeg(img, channels=3 if rgb else 1)
                    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
                    img = tf.image.convert_image_dtype(img, tf.float32)
                    # resize the image to the desired size.
                    return tf.image.resize(img, resolution)

                def process_path(file_path):
                    return tf.io.read_file(decode_img(file_path))

                for label, image_paths in label_images.items():
                    for image_path in image_paths:
                        key = '/'.join((label, os.path.basename(image_path)))
                        # temp_f = path.join(tempdir, '_'.join((label, os.path.basename(image_path))))

                        yield key, {
                            "image": process_path(image_path),
                            "label": label,
                        }

        builder = DrSpocImageLabelFolder(
            dataset_name=dataset_name,
            data_dir=data_dir
            # config=tfds.core.BuilderConfig(
            #     name='DR SPOC {}'.format(dataset_name[len('dr_spoc_'):]),
            #     version=tfds.core.Version('2019.12.28'),
            #     description='Coming soon'
            # )
        )

        return builder

    return builder_factory, data_dir, manual_dir


dr_spoc_builder.t = 1


def _get_manual_dir(dr_spoc_parent_dir, manual_dir):  # type: (str, str) -> str
    if path.dirname(manual_dir) != 'DR SPOC Dataset' \
        and not path.isdir(path.join(manual_dir, 'DR SPOC')) \
        and not path.isdir(path.join(path.dirname(manual_dir), 'DR SPOC')):
        symlinked_datasets_directory = path.join(dr_spoc_parent_dir,
                                                 'symlinked_datasets')
        manual_dir = symlinked_datasets_directory
    return manual_dir
