from argparse import ArgumentParser, FileType
from sys import modules, stdin

from numpy import float32

from ml_glaucoma import __version__
from ml_glaucoma.CNN import bmes_cnn
from ml_glaucoma.download import download
from ml_glaucoma.parser import parser

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

    '''
    data_parser = subparsers.add_parser('data', help='Data preprocessing runner')
    data_parser.add_argument('-s', '--save', help='Save h5 file of dataset, following preprocessing',
                             dest='preprocess_to', required=True)
    data_parser.add_argument('-f', '--force', dest='force_new', help='Force new h5 file of dataset being created',
                             action='store_true')
    data_parser.add_argument('-p', '--pixels', help='Pixels. E.g.: 400 for 400px * 400px',
                             type=int, default=400)
    '''

    download_parser = subparsers.add_parser('download', help='Download required data')
    download_parser.add_argument('-d', '--download-dir', help='Directory to store precompiled CNN nets', required=True)
    download_parser.add_argument('-f', '--force', dest='force_new', help='Force recreation of precompiled CNN nets',
                                 action='store_true')

    cnn_parser = subparsers.add_parser('cnn', help='Convolutional Neural Network runner')
    cnn_parser.add_argument('-b', '--batch-size', help='Batch size', default=128, type=int)
    cnn_parser.add_argument('-n', '--num-classes', help='Number of classes', default=2, type=int)
    cnn_parser.add_argument('-e', '--epochs', help='Number of epochs', type=float32)
    cnn_parser.add_argument('-m', '--model', help='Filename for h5 trained model file',
                            dest='model_name', required=True)
    cnn_parser.add_argument('-s', '--save', help='Save h5 file of dataset, following preprocessing',
                            dest='preprocess_to', required=True)
    cnn_parser.add_argument('-d', '--download-dir', help='Directory to store precompiled CNN nets', required=True)
    cnn_parser.add_argument('-t', '--transfer-model',
                            help='Transfer model. Currently any one of: `keras.application`, e.g.: "vgg16"; "resnet50"')
    cnn_parser.add_argument('--dropout', help='Dropout (0,1,2,3 or 4)', default=4, type=int)
    cnn_parser.add_argument('-p', '--pixels', help='Pixels. E.g.: 400 for 400px * 400px',
                            type=int, default=400)
    cnn_parser.add_argument('--tensorboard-log-dir', help='Enabled Tensorboard integration and sets its log dir')
    cnn_parser.add_argument('--optimizer', default='Adadelta')
    cnn_parser.add_argument('--loss', default='categorical_crossentropy')
    cnn_parser.add_argument('--architecture', help='Current options: unet; for U-Net architecture')
    cnn_parser.add_argument('--metrics', help='precision_recall or btp')
    cnn_parser.add_argument('--split-dir', help='Place to create symbolic links for train, test, validation split')
    cnn_parser.add_argument('--bmes123-pardir', help='Parent folder of BMES123 folder')
    cnn_parser.add_argument('--class-mode', help='Determines the type of label arrays that are returned',
                            choices=('categorical', 'binary', 'sparse'), default='categorical')

    post_parser = subparsers.add_parser('parser',
                                        help='Show metrics from output. Default: per epoch sensitivity & specificity.')
    post_parser.add_argument('infile', nargs='?', type=FileType('r'), default=stdin,
                             help='File to work from. Defaults to stdin. So can pipe.')
    post_parser.add_argument('--threshold', help='E.g.: 0.7 for sensitivity & specificity >= 70%%', type=float)
    post_parser.add_argument('--top', help='Show top k results', type=int)
    post_parser.add_argument('--by-diff', help='Sort by lowest difference between sensitivity & specificity',
                             action='store_true')

    return parser


if __name__ == '__main__':
    kwargs = dict(_build_parser().parse_args()._get_kwargs())

    command = kwargs.pop('command')

    if command is None:
        raise ReferenceError('You must specify a command. Append `--help` for details.')

    ({  # 'data': prepare_data,
      'download': download,
      'cnn': bmes_cnn.run,
      'parser': parser
      }[command])(**kwargs)
