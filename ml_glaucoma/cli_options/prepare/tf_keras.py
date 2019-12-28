from os import path

import tensorflow_datasets as tfds

from ml_glaucoma import problems as p


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

                if manual_dir is None:
                    raise ValueError(
                        '`manual_dir` must be provided if doing bmes_init')

                if dr_spoc_parent_dir is None:
                    raise ValueError(
                        '`dr_spoc_parent_dir` must be provided if '
                        '`dr_spoc_init is True`')

                get_data(dr_spoc_parent_dir, manual_dir)

            # builder = tfds.image.ImageLabelFolder('DR SPOC')
            # builder.as_dataset(split=('test', 'train', 'valid'), shuffle_files=False)

            builder = tfds.image.ImageLabelFolder('DR SPOC')

            dl_config = tfds.download.DownloadConfig(manual_dir=manual_dir)

            # manual_dir = path.join(bmes_parent_dir, 'tensorflow_datasets')

            builder.download_and_prepare(download_config=dl_config)
            # print(builder.info)  # Splits, num examples,... automatically extracted
            # ds = builder.as_dataset(split=('test', 'train', 'valid'), shuffle_files=True)
            builders.append(builder)
            # TODO: Ensure resolution, RGB, and data_dir can be provided like the other datasets
            return
        else:
            raise NotImplementedError()

        builder = builder_factory(
            resolution=resolution,
            rgb=not gray_on_disk,
            data_dir=data_dir)

        p.download_and_prepare(
            builder=builder,
            download_config=tfds.download.DownloadConfig(
                extract_dir=extract_dir, manual_dir=manual_dir,
                download_mode=download_mode),
            download_dir=download_dir)
        builders.append(builder)
