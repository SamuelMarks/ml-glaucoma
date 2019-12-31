import itertools
import os
from os import path

import tensorflow_datasets as tfds
from tensorflow_datasets.image.image_folder import list_folders, list_imgs

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
            def _split_generators(self, dl_manager):
                """Returns SplitGenerators from the folder names."""
                # At data creation time, parse the folder to deduce number of splits,
                # labels, image size,

                # The splits correspond to the high level folders
                split_names = list_folders(dl_manager.manual_dir)

                # Extract all label names and associated images
                split_label_images = {}  # dict[split_name][label_name] = list(img_paths)
                for split_name in split_names:
                    split_dir = os.path.join(dl_manager.manual_dir, split_name)
                    split_label_images[split_name] = {
                        label_name: list_imgs(os.path.join(split_dir, label_name))
                        for label_name in list_folders(split_dir)
                    }

                # Merge all label names from all splits to get the final list of labels
                # Sorted list for determinism
                labels = [split.keys() for split in split_label_images.values()]
                labels = list(sorted(set(itertools.chain(*labels))))

                # Could improve the automated encoding format detection
                # Extract the list of all image paths
                image_paths = [
                    image_paths
                    for label_images in split_label_images.values()
                    for image_paths in label_images.values()
                ]
                if any(f.lower().endswith(".png") for f in itertools.chain(*image_paths)):
                    encoding_format = "png"
                else:
                    encoding_format = "jpeg"

                # Update the info.features. Those info will be automatically resored when
                # the dataset is re-created
                self.info.features["image"].set_encoding_format(encoding_format)

                self.info.features["image"].set_shape(resolution + ((3 if rgb else 1),))
                self.info.features["label"].names = labels

                def num_examples(label_images):
                    return sum(len(imgs) for imgs in label_images.values())

                # Define the splits
                return [
                    tfds.core.SplitGenerator(
                        name=split_name,
                        # The number of shards is a dynamic function of the total
                        # number of images (between 0-10)
                        num_shards=min(10, max(num_examples(label_images) // 1000, 1)),
                        gen_kwargs=dict(label_images=label_images, ),
                    ) for split_name, label_images in split_label_images.items()
                ]

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
