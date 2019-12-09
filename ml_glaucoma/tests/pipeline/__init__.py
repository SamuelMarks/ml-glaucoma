from json import dumps
from os import path, environ, remove
from tempfile import NamedTemporaryFile
from unittest import TestCase, main as unittest_main

from pkg_resources import resource_filename

from ml_glaucoma.cli_options.parser import cli_handler


class TestPipeline(TestCase):
    def setUp(self):
        self.logfile = NamedTemporaryFile('pipeline', delete=False)

    def tearDown(self):
        remove(self.logfile)

    def test_simple(self):
        cli_handler([
            'pipeline',
            '--options', dumps({'losses': [{'BinaryCrossentropy': 0}, {'JaccardDistance': 0}],
                                'models': [{'DenseNet169': 0}, {"EfficientNetB0": 0}],
                                "optimizers": [{"Adadelta": 0}, {"Adagrad": 0}, {"Adam": 0}]}),
            '--key', 'models',
            '--threshold', '100',
            '--logfile', self.logfile,

            'train',
            '-ds', 'refuge',
            '--data_dir', environ.get('DATA_DIR', '/mnt/tensorflow_datasets'),
            '--model_file', path.join(path.dirname(resource_filename('ml_glaucoma', "__init__.py")), "model_configs",
                                      "applications.gin"),
            '--model_dir', environ.get('MODEL_DIR',
                                       '/mnt/ml_glaucoma_models/'
                                       'gon_DenseNet169_Adam_BinaryCrossentropy_epochs_250_again036'),
            '--model_param', 'application = "DenseNet169"',
            '--epochs', '250',
            '--delete-lt', '0.96',
            '--losses', 'BinaryCrossentropy',
            '--optimizers', 'Adadelta',
            '--tensorboard_log_dir', environ.get('MODEL_DIR', '/mnt/ml_glaucoma_models/'
                                                              'gon_DenseNet169_Adam_BinaryCrossentropy_epochs_250_again036'),
            '--models', 'DenseNet169'])


if __name__ == '__main__':
    unittest_main()
