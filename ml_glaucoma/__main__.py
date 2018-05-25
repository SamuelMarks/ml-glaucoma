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
    parser = ArgumentParser(description='CLI for a Glaucoma diagnosing CNN and preparing data for such')
    parser.add_argument('--version', action='version',
                        version='{} {}'.format(modules[__name__].__package__, __version__))

    subparsers = parser.add_subparsers(dest='command')

    data_parser = subparsers.add_parser('data', help='Data preprocessing runner')
    data_parser.add_argument('-s', '--save', dest='save_to', help='Save h5 file of dataset', required=True)
    data_parser.add_argument('-f', '--force', dest='force_new', help='Force new h5 file of dataset being created',
                             action='store_true')

    download_parser = subparsers.add_parser('download', help='Download required data')
    download_parser.add_argument('-d', '--download-dir', help='Directory to store precompiled CNN nets', required=True)
    download_parser.add_argument('-f', '--force', dest='force_new', help='Force recreation of precompiled CNN nets',
                                 action='store_true')

    cnn_parser = subparsers.add_parser('cnn', help='Convolutional Neural Network runner')
    cnn_parser.add_argument('-b', '--batch-size', help='Batch size', default=128, type=int)
    cnn_parser.add_argument('-n', '--num-classes', help='Number of classes', default=2, type=int)
    cnn_parser.add_argument('-e', '--epochs', help='Number of epochs', default=20, type=int)
    cnn_parser.add_argument('-m', '--model', help='Filename for h5 trained model file',
                            dest='model_name', required=True)
    cnn_parser.add_argument('-s', '--save', help='Save h5 file of dataset', dest='save_to',
                            required=True)
    cnn_parser.add_argument('-d', '--download-dir', help='Directory to store precompiled CNN nets', required=True)
    cnn_parser.add_argument('-t', '--transfer-model', help='Transfer model. Currently one of: "vgg16"; "resnet50"')

    return parser


if __name__ == '__main__':
    kwargs = dict(_build_parser().parse_args()._get_kwargs())

    command = kwargs.pop('command')
    getattr(bmes_cnn, {
        'data': 'prepare_data',
        'download': 'download',
        'cnn': 'run'
    }[command])(**kwargs)
