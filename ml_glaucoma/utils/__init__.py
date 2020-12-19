from collections import deque, namedtuple
from datetime import datetime
from itertools import islice
from os import environ
from platform import python_version_tuple
from pprint import PrettyPrinter

redis_cursor = None
if "NO_REDIS" not in environ:
    from redis import StrictRedis

    redis_cursor = StrictRedis(host="localhost", port=6379, db=0)

if python_version_tuple()[0] == "3":
    xrange = range

pp = PrettyPrinter(indent=4).pprint

it_consumes = (
    lambda it, n=None: deque(it, maxlen=0)
    if n is None
    else next(islice(it, n, n), None)
)


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, datetime):
        serial = obj.isoformat()
        return serial
    raise TypeError("Type not serializable")


def find_nth(s, x, n=0, overlap=False):
    l = 1 if overlap else len(x)
    i = -l
    for c in xrange(n + 1):
        i = s.find(x, i + l)
        if i < 0:
            break
    return i


def sorted_enumerate(seq):
    return tuple(i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq)))


def obj_to_d(obj):
    return (
        obj
        if type(obj) is dict
        else dict((k, getattr(obj, k)) for k in dir(obj) if not k.startswith("_"))
    )


def update_d(d, arg=None, **kwargs):
    if arg:
        d.update(arg)
    if kwargs:
        d.update(kwargs)
    return d


def namedtuple2dict(nt):  # type: (namedtuple) -> dict
    return dict(nt._asdict())


# From https://rosettacode.org/wiki/Longest_common_subsequence#Dynamic_Programming_8
def lcs(a, b):
    # generate matrix of length of longest common subsequence for substrings of both words
    lengths = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

    # read a substring from the matrix
    result = ""
    j = len(b)
    for i in range(1, len(a) + 1):
        if lengths[i][j] != lengths[i - 1][j]:
            result += a[i - 1]

    return result


del namedtuple, datetime, islice, environ, python_version_tuple, PrettyPrinter

__all__ = [
    "pp",
    "it_consumes",
    "run_once",
    "json_serial",
    "find_nth",
    "sorted_enumerate",
    "obj_to_d",
    "namedtuple2dict",
    "lcs",
]
