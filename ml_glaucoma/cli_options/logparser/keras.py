from functools import reduce
from itertools import groupby, takewhile, islice
from operator import itemgetter
from platform import python_version_tuple

from six import iteritems

from ml_glaucoma import get_logger
from ml_glaucoma.utils import pp

if python_version_tuple()[0] == '3':
    imap = map
    ifilter = filter

logger = get_logger(__file__)


def log_parser(infile, top, threshold, by_diff):
    # This one is useful for simple Keras output
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
