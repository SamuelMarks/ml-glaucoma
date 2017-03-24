from collections import namedtuple

from os import makedirs, mkdir, path
from redis import StrictRedis
import cPickle as pickle

from ml_glaucoma.utils import redis_cursor, it_consumes


class Cache(object):
    """Example:
    import random

    pickled_cache = Cache.update_locals(Cache.load(), locals())

    if 'r' not in pickled_cache:
        pickled_cache['r'] = r = random.random()

    Cache.save(pickled_cache)
    print 'r =', r"""

    __slots__ = ('load', 'save', 'update_locals', 'fname')

    def __init__(self, fname=redis_cursor):
        if isinstance(fname, StrictRedis):
            self.fname = fname
        else:
            self.fname = fname or path.join(path.dirname(path.dirname(__file__)), '_data', '.cache',
                                            'pickled_cache.pkl')
            directory = path.dirname(fname)
            if not path.isdir(directory):
                makedirs(directory)
            open(self.fname, 'a').close()

    def load(self, fname=None, key=None, marshall=pickle):
        print '_cache_load called'

        self.fname = fname or self.fname
        if isinstance(self.fname, StrictRedis):
            res = self.fname.get('.pycache/{key}'.format(key=key if key is not None else 'caches.pkl'))
            _pickled_cache = marshall.loads(res) if res is not None else {}
        elif path.isfile(self.fname) and path.getsize(self.fname):
            with open(self.fname, 'rb') as f:
                _pickled_cache = marshall.load(f)
        else:
            _pickled_cache = {}

        return _pickled_cache

    @staticmethod
    def update_locals(_pickled_cache, locls):
        print '_update_locals called'
        print '+len(_pickled_cache) =', len(_pickled_cache)
        if type(_pickled_cache) is dict:
            it_consumes(locls.__setitem__(k, v)
                        for k, v in _pickled_cache.iteritems())
        else:
            raise TypeError('_pickled cache isn\'t dict')
        # elif key is not None: locls[key] = _pickled_cache

        print ':len(_pickled_cache) =', len(_pickled_cache)
        return _pickled_cache

    def save(self, cache, fname=None, key=None, marshall=pickle):
        self.fname = fname or self.fname

        if isinstance(self.fname, StrictRedis):
            print 'saving cache to redis://{}'.format(
                '.pycache/{key}'.format(key=key if key is not None else 'caches.pkl'))
            self.fname.set('.pycache/{key}'.format(key=key if key is not None else 'caches.pkl'), marshall.dumps(cache))
        else:
            print 'saving cache to file://{}'.format(self.fname)
            with open(self.fname, 'wb') as f:
                marshall.dump(cache, f)


# TODO: Make another function which calls this to decorate
def eval_cache(cache, key, call, replace=False, *args, **kwargs):
    assert callable(call)
    if key not in cache or replace:
        cache[key] = call(*args, **kwargs)
    return cache[key]
