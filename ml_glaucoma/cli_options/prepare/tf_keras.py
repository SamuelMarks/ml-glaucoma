import tensorflow_datasets as tfds

from ml_glaucoma import problems as p


def dataset_builder(bmes_init, bmes_parent_dir, builders, data_dir, dataset, download_dir, download_mode,
                    extract_dir, gray_on_disk, manual_dir, resolution):
    for ds in set(dataset):  # remove duplicates
        if ds == 'bmes':
            from ml_glaucoma.tfds_builders import bmes

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
            from ml_glaucoma.tfds_builders import refuge

            builder_factory = refuge.get_refuge_builder
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
