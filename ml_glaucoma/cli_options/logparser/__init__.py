from argparse import FileType
from os import environ
from sys import stdin

from ml_glaucoma.cli_options.base import Configurable

if environ['TF']:
    from ml_glaucoma.cli_options.logparser.tf import log_parser
elif environ['TORCH']:
    from ml_glaucoma.cli_options.logparser.torch import log_parser
else:
    from ml_glaucoma.cli_options.logparser.other import log_parser


class ConfigurableLogParser(Configurable):
    description = 'Parse out metrics from log output. Default: per epoch sensitivity & specificity.'

    def fill_self(self, parser):
        parser.description = 'Show metrics from output. Default: per epoch sensitivity & specificity.'
        parser.add_argument('infile', nargs='?', type=FileType('r'), default=stdin,
                            help='File to work from. Defaults to stdin. So can pipe.')
        parser.add_argument('-d', '--directory', help='Directory. Searches here rather than infile.')
        parser.add_argument('--threshold', help='E.g.: 0.7 for sensitivity & specificity >= 70%%', default='0.7',
                            type=float)
        parser.add_argument('--top', help='Show top k results', default=5, type=int)
        parser.add_argument('--by-diff', help='Sort by lowest difference between sensitivity & specificity',
                            action='store_true')
        parser.add_argument('--tag', help='Tag to filter by', default='epoch_val_auc')

    def build(self, **kwargs):
        return self.build_self(**kwargs)

    def build_self(self, **kwargs):
        return log_parser(**kwargs)


__all__ = ['ConfigurableLogParser']
