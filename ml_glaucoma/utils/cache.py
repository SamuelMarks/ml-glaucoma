from collections import namedtuple

from os import makedirs, mkdir, path
from redis import StrictRedis
import cPickle as pickle

from ml_glaucoma.utils import redis_cursor, it_consumes


def _cache_init(fname=redis_cursor):
    if fname is not None:
        if isinstance(fname, StrictRedis):
            return fname
        dir_ = path.dirname(fname)
        if not path.isdir(dir_):
            makedirs(dir_)
        fname = fname or path.join('.cache', 'pickled_cache.pkl')
        open(fname, 'a').close()
    # elif not path.isdir('.cache'): mkdir('.cache')

    return fname


def _cache_load(fname=redis_cursor, key=None, marshall=pickle):
    fname = _cache_init(fname)

    if isinstance(fname, StrictRedis):
        res = fname.get('.pycache/{key}'.format(key=key if key is not None else 'caches.pkl'))
        _pickled_cache = marshall.loads(res) if res is not None else {}
    elif path.isfile(fname) and path.getsize(fname):
        with open(fname, 'rb') as f:
            _pickled_cache = marshall.load(f)
    else:
        _pickled_cache = {}

    return _pickled_cache


def _update_locals(_pickled_cache, locls):
    if type(_pickled_cache) is dict:
        it_consumes(locls.__setitem__(k, v)
                    for k, v in _pickled_cache.iteritems())
    else:
        raise TypeError('_pickled cache isn\'t dict')
    # elif key is not None: locls[key] = _pickled_cache

    return _pickled_cache


def _cache_save(cache, fname=redis_cursor, key=None, marshall=pickle):
    fname = _cache_init(fname)

    if isinstance(fname, StrictRedis):
        fname.set('.pycache/{key}'.format(key=key if key is not None else 'caches.pkl'), marshall.dumps(cache))
    else:
        with open(fname, 'wb') as f:
            marshall.dump(cache, f)


# TODO: Make another function which calls this to decorate
def eval_cache(cache, key, call, replace=False, *args, **kwargs):
    assert callable(call)
    if key not in cache or replace:
        cache[key] = call(*args, **kwargs)
    return cache[key]


Cache = namedtuple('Cache', ('save', 'load', 'update_locals'))(
    save=_cache_save, load=_cache_load, update_locals=_update_locals)
# Cache.__doc__ =
"""Example:
import random

pickled_cache = Cache.update_locals(Cache.load(), locals())

if 'r' not in pickled_cache:
    pickled_cache['r'] = r = random.random()

Cache.save(pickled_cache)
print 'r =', r"""
