from os import path

import tensorflow_datasets as tfds

from ml_glaucoma import problems as p, get_logger
from ml_glaucoma.datasets.tfds_builders.dr_spoc import dr_spoc_builder, dr_spoc_datasets_set

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

        elif ds in dr_spoc_datasets_set:  # 'DR SPOC', 'DR SPOC - grad_and_no_grad_dir', 'DR SPOC - no_no_grad_dir'
            builder_factory, data_dir, manual_dir = dr_spoc_builder(ds, data_dir, dr_spoc_init,
                                                                    dr_spoc_parent_dir, manual_dir)
        else:
            raise NotImplementedError()

        builder = builder_factory(resolution=resolution,
                                  rgb=not gray_on_disk,
                                  data_dir=data_dir)

        if dataset_builder.t > 0:
            dataset_builder.t -= 1
            print('download_dir:'.ljust(20), '{!r}\n'.format(download_dir),
                  'data_dir:'.ljust(20), '{!r}\n'.format(data_dir),
                  'manual_dir:'.ljust(20), '{!r}\n'.format(manual_dir),
                  sep='')

        p.download_and_prepare(
            builder=builder,
            download_config=tfds.download.DownloadConfig(
                extract_dir=extract_dir, manual_dir=manual_dir,
                download_mode=download_mode
            ),
            download_dir=download_dir
        )

        builders.append(builder)


dataset_builder.t = 1
