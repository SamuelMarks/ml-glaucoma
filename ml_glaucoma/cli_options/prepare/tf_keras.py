from os import path

import tensorflow_datasets as tfds

from ml_glaucoma import problems as p, get_logger

logger = get_logger('.'.join((path.basename(path.dirname(__file__)),
                              path.basename(__file__).rpartition('.')[0])))


def dataset_builder(dataset, data_dir, download_dir,
                    extract_dir, manual_dir, download_mode,
                    resolution, gray_on_disk,
                    bmes_init, bmes_parent_dir,
                    dr_spoc_init, dr_spoc_parent_dir,
                    builders):
    for ds in frozenset(dataset):  # remove duplicates
        if ds == 'bmes':
            from ml_glaucoma.datasets.tfds_builders import bmes

            builder_factory = bmes.get_bmes_builder
            if bmes_init:
                from ml_glaucoma.utils.bmes_data_prep import get_data

                if manual_dir is None:
                    raise ValueError(
                        '`manual_dir` must be provided if doing bmes_init')

                if bmes_parent_dir is None:
                    raise ValueError(
                        '`bmes_parent_dir` must be provided if doing '
                        'bmes_init')

                get_data(bmes_parent_dir, manual_dir)
        elif ds == 'refuge':
            from ml_glaucoma.datasets.tfds_builders import refuge

            builder_factory = refuge.get_refuge_builder
        elif ds == 'dr_spoc':
            if dr_spoc_init:
                from ml_glaucoma.utils.dr_spoc_data_prep import get_data

                if dr_spoc_parent_dir is None:
                    raise ValueError(
                        '`dr_spoc_parent_dir` must be provided if '
                        '`dr_spoc_init is True`')

                get_data(root_directory=dr_spoc_parent_dir, manual_dir=manual_dir)

            part = 'tensorflow_datasets'
            if not data_dir.endswith(part):
                data_dir = path.join(data_dir, part)

            builder = tfds.image.ImageLabelFolder(
                'DR SPOC Photo Dataset', data_dir=data_dir,
                config=tfds.core.BuilderConfig(
                    name='DR SPOC',
                    version='2019.12.28',
                    description='Coming soon'
                )
            )

            # manual_dir = path.join(bmes_parent_dir, 'tensorflow_datasets')

            # print(builder.info)  # Splits, num examples,... automatically extracted
            # ds = builder.as_dataset(split=('test', 'train', 'valid'), shuffle_files=True)
            # builders.append(builder)
            #
            # return

            # print('ml_glaucoma/cli_options/prepare/tf_keras.py::data_dir: {!r}'.format(data_dir))

            # TODO: Ensure resolution, RGB can be provided
            def builder_factory(resolution, rgb, data_dir):
                if resolution is not None:
                    logger.warn('`resolution` not handled (yet) for DR SPOC dataset')
                if rgb is not None:
                    logger.warn('`rgb` not handled (yet) for DR SPOC dataset')
                builder._data_dir = data_dir
                return builder

        else:
            raise NotImplementedError()

        builder = builder_factory(resolution=resolution,
                                  rgb=not gray_on_disk,
                                  data_dir=data_dir)

        p.download_and_prepare(
            builder=builder,
            download_config=tfds.download.DownloadConfig(
                extract_dir=extract_dir, manual_dir=manual_dir,
                download_mode=download_mode
            ),
            download_dir=download_dir
        )
        builders.append(builder)
