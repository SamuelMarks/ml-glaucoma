from argparse import ArgumentParser
from sys import modules

from ml_glaucoma import __version__
from ml_glaucoma.CNN import bmes_cnn

# Original options
'''
batch_size = 128
num_classes = 2
epochs = 20
DATA_SAVE_LOCATION = '/mnt/datasets/400x400balanced_dataset.hdf5'
save_dir = path.join(getcwd(), 'saved_models')
model_name = 'keras_glaucoma_trained_model.h5'
'''

def _build_parser():
    parser = ArgumentParser(description='CLI for a Glaucoma diagnosing CNN')
    parser.add_argument('-b', '--batch-size', help='Batch size', default=128)
    parser.add_argument('-n', '--num-classes', help='Number of classes', default=2)
    parser.add_argument('-e', '--epochs', help='Number of epochs', default=20)
    parser.add_argument('-m', '--model', help='Filename for h5 trained model file', dest='model_name', required=True)
    parser.add_argument('-o', '--output', help='Output in hdf5 format to this filename', required=True)
    parser.add_argument('--version', action='version',
                        version='{} {}'.format(modules[__name__].__package__, __version__))
    return parser


if __name__ == '__main__':
    kwargs = dict(_build_parser().parse_args()._get_kwargs())
    bmes_cnn.run(**kwargs)
