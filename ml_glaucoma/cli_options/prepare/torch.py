def dataset_builder(bmes_init, bmes_parent_dir, builders, data_dir, dataset, download_dir, download_mode,
                    extract_dir, gray_on_disk, manual_dir, resolution):
    for ds in frozenset(dataset):
        if ds == 'bmes':
            from ml_glaucoma.datasets.torch import bmes

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

            return bmes.get_bmes_builder(bmes_parent_dir)

        elif ds == 'refuge':
            raise NotImplementedError(ds)
        else:
            raise NotImplementedError(ds)
