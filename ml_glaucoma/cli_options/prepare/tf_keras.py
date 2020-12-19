from os import path

import tensorflow_datasets as tfds

from ml_glaucoma import get_logger
from ml_glaucoma import problems as p
from ml_glaucoma.datasets.tfds_builders.bmes import bmes_builder
from ml_glaucoma.datasets.tfds_builders.dr_spoc import (
    dr_spoc_builder,
    dr_spoc_datasets_set,
)

logger = get_logger(
    ".".join(
        (
            path.basename(path.dirname(__file__)),
            path.basename(__file__).rpartition(".")[0],
        )
    )
)


def dataset_builder(
    dataset,
    data_dir,
    download_dir,
    extract_dir,
    manual_dir,
    download_mode,
    resolution,
    gray_on_disk,
    bmes_init,
    bmes_parent_dir,
    dr_spoc_init,
    dr_spoc_parent_dir,
    builders,
    force_create=False,
):
    for ds in frozenset(dataset):  # remove duplicates
        if ds == "bmes":
            builder_factory, data_dir, manual_dir = bmes_builder(
                data_dir=data_dir,
                init=bmes_init,
                parent_dir=bmes_parent_dir,
                manual_dir=manual_dir,
                force_create=force_create,
            )
        elif ds == "refuge":
            from ml_glaucoma.datasets.tfds_builders import refuge

            builder_factory = refuge.get_refuge_builder

        elif (
            ds in dr_spoc_datasets_set
        ):  # 'DR SPOC', 'DR SPOC - grad_and_no_grad_dir', 'DR SPOC - no_no_grad_dir'
            builder_factory, data_dir, manual_dir = dr_spoc_builder(
                dataset_name=ds,
                data_dir=data_dir,
                dr_spoc_init=dr_spoc_init,
                dr_spoc_parent_dir=dr_spoc_parent_dir,
                manual_dir=manual_dir,
                force_create=force_create,
            )
        else:
            raise NotImplementedError(ds)

        builder = builder_factory(
            resolution=resolution, rgb=not gray_on_disk, data_dir=data_dir
        )

        if dataset_builder.t > 0:
            dataset_builder.t -= 1
            print(
                "data_dir:".ljust(20),
                "{!r}\n".format(data_dir),
                "extract_dir:".ljust(20),
                "{!r}\n".format(extract_dir),
                "manual_dir:".ljust(20),
                "{!r}\n".format(manual_dir),
                "download_dir:".ljust(20),
                "{!r}\n".format(download_dir),
                sep="",
            )

        p.download_and_prepare(
            builder=builder,
            download_config=tfds.download.DownloadConfig(
                extract_dir=extract_dir,
                manual_dir=manual_dir,
                download_mode=download_mode,
            ),
            download_dir=download_dir,
        )

        builders.append(builder)


dataset_builder.t = 1

__all__ = ["dataset_builder"]
