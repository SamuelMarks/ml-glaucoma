from __future__ import print_function

from argparse import FileType
from sys import stdin

from ml_glaucoma.cli_options.base import Configurable

from itertools import islice, groupby, takewhile
from operator import itemgetter
from platform import python_version_tuple

from six import iteritems

from ml_glaucoma import get_logger

if python_version_tuple()[0] == '3':
    from functools import reduce

    imap = map
    ifilter = filter

from ml_glaucoma.utils import pp

logger = get_logger(__file__)


def log_parser(infile, top, threshold, by_diff):
    epoch2stat = {
        key: val
        for key, val in iteritems(
            {
                k: tuple(imap(itemgetter(1), v))
                for k, v in groupby(imap(lambda l: (l[0], l[1]),
                                         ifilter(None, imap(
                                             lambda l: (lambda fst: (
                                                 lambda three: (int(three), l.rstrip()[l.rfind(':') + 2:])
                                                 if three is not None and three.isdigit() and int(
                                                     three[0]) < 4 else None)(
                                                 l[fst - 3:fst] if fst > -1 else None))(l.rfind(']')), infile)
                                                 ))
                                    , itemgetter(0))
            })
        if val and len(val) == 2
    }

    if threshold is not None:
        within_threshold = sorted((
            (k, reduce(lambda a, b: a >= threshold <= b, imap(
                lambda val: float(''.join(takewhile(lambda c: c.isdigit() or c == '.', val[::-1]))[::-1]),
                v))
             ) for k, v in iteritems(epoch2stat)), key=itemgetter(1))
        pp(tuple(islice((epoch2stat[k[0]] for k in within_threshold if k[1]), 0, top)))
    elif by_diff:
        lowest_diff = sorted((
            (k, reduce(lambda a, b: abs(a - b),
                       imap(lambda val: float(''.join(takewhile(lambda c: c.isdigit() or c == '.', val[::-1]))[::-1]),
                            v))
             ) for k, v in iteritems(epoch2stat)), key=itemgetter(1))

        pp(tuple(islice((epoch2stat[k[0]] for k in lowest_diff), 0, top)))
    else:
        pp(epoch2stat)


class ConfigurableLogParser(Configurable):
    description = 'Parse out metrics from log output. Default: per epoch sensitivity & specificity.'

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
