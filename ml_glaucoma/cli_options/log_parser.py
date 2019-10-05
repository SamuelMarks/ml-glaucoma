from argparse import FileType
from sys import stdin

from ml_glaucoma.cli_options.base import Configurable
from ml_glaucoma.log_parser import log_parser


class ConfigurableLogParser(Configurable):
    description = 'Parse out metrics from log output. Default: per epoch sensitivity & specificity.'

    def __init__(self, infile=None, threshold=None, top=None, by_diff=None):
        super(ConfigurableLogParser, self).__init__(
            infile=infile, threshold=threshold, top=top, by_diff=by_diff
        )

    def fill_self(self, parser):
        parser.description = 'Show metrics from output. Default: per epoch sensitivity & specificity.'
        parser.add_argument('infile', nargs='?', type=FileType('r'), default=stdin,
                            help='File to work from. Defaults to stdin. So can pipe.')
        parser.add_argument('--threshold', help='E.g.: 0.7 for sensitivity & specificity >= 70%%', default='0.7',
                            type=float)
        parser.add_argument('--top', help='Show top k results', default=5, type=int)
        parser.add_argument('--by-diff', help='Sort by lowest difference between sensitivity & specificity',
                            action='store_true')

    def build(self, **kwargs):
        return self.build_self(**kwargs)

    def build_self(self, infile, threshold, top, by_diff, **kwargs):
        return log_parser(
            infile=infile,
            threshold=threshold,
            top=top,
            by_diff=by_diff
        )
