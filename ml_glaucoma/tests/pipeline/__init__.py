import sys
from contextlib import contextmanager
from json import dumps, loads
from os import path, environ, remove, rmdir
from tempfile import mkdtemp
from unittest import TestCase, main as unittest_main

from pkg_resources import resource_filename
from six import StringIO

from ml_glaucoma.cli_options.parser import cli_handler


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class TestPipeline(TestCase):
    def setUp(self):
        self.tempdir = mkdtemp(prefix='pipeline')
        self.logfile = path.join(self.tempdir, 'pipeline.log')

    def tearDown(self):
        remove(self.logfile)
        rmdir(self.tempdir)

    def test_simple(self):
        threshold = 100

        with captured_output() as (out, err):
            cli_handler([
                'pipeline',
                '--options', dumps({'losses': [{'BinaryCrossentropy': 0}, {'JaccardDistance': 0}],
                                    'models': [{'DenseNet169': 0}, {'EfficientNetB0': 0}],
                                    'optimizers': [{'Adadelta': 0}, {'Adagrad': 0}, {'Adam': 0}]}),
                '--key', 'models',
                '--threshold', '{}'.format(threshold),
                '--logfile', self.logfile,
                '--dry-run',

                'train',
                '-ds', 'refuge',
                '--data_dir', environ.get('DATA_DIR', '/mnt/tensorflow_datasets'),
                '--model_file',
                path.join(path.dirname(resource_filename('ml_glaucoma', '__init__.py')), 'model_configs',
                          'applications.gin'),
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
                '--models', 'DenseNet169'
            ])
        err.seek(0)
        out.seek(0)
        all_options = tuple(
            map(lambda options: cli_handler(cmd=options, return_namespace=True),
                map(loads,
                    map(lambda line: line[len('rest:'):-1].lstrip(),
                        filter(lambda line: line.startswith('rest:     ['),
                               out.read().split('\n'))
                        )
                    )
                )
        )
        self.assertEqual(err.read(), '')
        self.assertEqual(len(all_options), threshold)


if __name__ == '__main__':
    unittest_main()
