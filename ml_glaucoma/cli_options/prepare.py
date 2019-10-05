from ml_glaucoma import problems as p
from ml_glaucoma.cli_options.base import Configurable


class ConfigurableBuilders(Configurable):
    def __init__(self):
        super(ConfigurableBuilders, self).__init__()

    def fill_self(self, parser):
        parser.add_argument(
            '-ds', '--dataset', choices=('bmes', 'refuge'), default=['refuge'],
            nargs='+',
            help='dataset key', )
        parser.add_argument(
            '--data_dir',
            help='root directory to store processed tfds records')
        parser.add_argument(
            '--download_dir', help='directory to store downloaded files')
        parser.add_argument(
            '--extract_dir', help='directory where extracted files are stored')
        parser.add_argument(
            '--manual_dir',
            help='directory where manually downloaded files are saved')
        parser.add_argument(
            '--download_mode',
            choices=(
                'reuse_dataset_if_exists',
                'reuse_cache_if_exists',
                'force_redownload'),
            help='tfds.GenerateMode')
        parser.add_argument(
            '-r', '--resolution', nargs=2, default=(256, 256), type=int,
            help='image resolution')
        parser.add_argument(
            '--gray_on_disk', action='store_true',
            help='whether or not to save data as grayscale on disk')
        parser.add_argument(
            '--bmes_init', action='store_true', help='initial bmes get_data')
        parser.add_argument(
            '--bmes_parent_dir', help='parent directory of bmes data')

    def build_self(self, dataset, data_dir, download_dir, extract_dir, manual_dir,
                   download_mode, resolution, gray_on_disk, bmes_init, bmes_parent_dir, **kwargs):
        import tensorflow_datasets as tfds
        builders = []
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
                raise NotImplementedError

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

        return builders
