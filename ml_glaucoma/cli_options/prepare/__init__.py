from os import environ

from ml_glaucoma.cli_options.base import Configurable

if environ['TF']:
    from ml_glaucoma.cli_options.prepare.tf_keras import dataset_builder
elif environ['TORCH']:
    from ml_glaucoma.cli_options.prepare.torch import dataset_builder
else:
    from ml_glaucoma.cli_options.prepare.other import dataset_builder


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
        builders = []

        dataset_builder(bmes_init, bmes_parent_dir, builders, data_dir, dataset, download_dir,
                        download_mode, extract_dir, gray_on_disk, manual_dir, resolution)

        return builders
