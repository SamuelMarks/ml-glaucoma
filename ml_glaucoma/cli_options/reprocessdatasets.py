from ml_glaucoma.cli_options.base import Configurable
from ml_glaucoma.cli_options.reprocess_datasets import reprocess_datasets


class ConfigurableReprocessDatasets(Configurable):
    description = "ReprocessDatasets subcommand"

    def __init__(self):
        super(ConfigurableReprocessDatasets, self).__init__()

    def fill_self(self, parser):
        parser.add_argument("filepaths", nargs="*", help="filepaths to tfrecord files")

    def build_self(self, filepaths, rest):
        print(
            "ReprocessDatasets::filepaths:",
            filepaths,
            "\nReprocessDatasets::rest:",
            rest,
        )
        return reprocess_datasets(filepaths)


__all__ = ["ConfigurableReprocessDatasets"]
