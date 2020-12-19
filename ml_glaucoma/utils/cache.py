from os import makedirs, path, remove
from platform import python_version_tuple
from sys import modules

from redis import StrictRedis
from six import iteritems

from ml_glaucoma import get_logger

if python_version_tuple()[0] == "2":
    import cPickle as pickle
else:
    import pickle

from ml_glaucoma.utils import it_consumes, redis_cursor

logger = get_logger(modules[__name__].__name__)


class Cache(object):
    """Example:
    import random

    pickled_cache = Cache.update_locals(cache.load(), locals())

    if 'r' not in pickled_cache:
        pickled_cache['r'] = r = random.random()

    Cache.save(pickled_cache)
    print 'r =', r"""

    def __init__(self, fname=redis_cursor):
        if isinstance(fname, StrictRedis):
            self.fname = fname
        else:
            self.fname = fname or path.join(
                path.dirname(path.dirname(__file__)),
                "_data",
                ".cache",
                "pickled_cache.pkl",
            )
            directory = path.dirname(fname)
            if not path.isdir(directory):
                makedirs(directory)
            try:
                open(self.fname, "a").close()
            except PermissionError:
                pass

    def load(self, fname=None, key=None, marshall=pickle):
        self.fname = fname or self.fname
        if isinstance(self.fname, StrictRedis):
            res = self.fname.get(
                ".pycache/{key}".format(key=key if key is not None else "caches.pkl")
            )
            _pickled_cache = marshall.loads(res) if res is not None else {}
        elif path.isfile(self.fname) and path.getsize(self.fname):
            with open(self.fname, "rb") as f:
                _pickled_cache = marshall.load(f)
        else:
            _pickled_cache = {}

        return _pickled_cache

    @staticmethod
    def update_locals(_pickled_cache, locls):
        if type(_pickled_cache) is dict:
            it_consumes(locls.__setitem__(k, v) for k, v in iteritems(_pickled_cache))
        else:
            raise TypeError("_pickled cache isn't dict")
        # elif key is not None: locls[key] = _pickled_cache

        return _pickled_cache

    def save(self, cache, fname=None, key=None, marshall=pickle):
        self.fname = fname or self.fname

        if isinstance(self.fname, StrictRedis):
            logger.info(
                "saving cache to redis://{}".format(
                    ".pycache/{key}".format(
                        key=key if key is not None else "caches.pkl"
                    )
                )
            )
            self.fname.set(
                ".pycache/{key}".format(key=key if key is not None else "caches.pkl"),
                marshall.dumps(cache),
            )
        else:
            logger.info("saving cache to file://{}".format(self.fname))
            with open(self.fname, "wb") as f:
                marshall.dump(cache, f)

    def invalidate(self, fname=None, key=None):
        self.fname = fname or self.fname

        if isinstance(self.fname, StrictRedis):
            key = key if key is not None else "caches.pkl"
            logger.info("rm redis://{}".format(".pycache/{key}".format(key=key)))
            self.fname.delete(".pycache/{key}".format(key=key))
        else:
            logger.info("rm {}".format(self.fname))
            remove(self.fname)


# TODO: Make another function which calls this to decorate
def eval_cache(cache, key, call, replace=False, *args, **kwargs):
    assert callable(call)
    if key not in cache or replace:
        cache[key] = call(*args, **kwargs)
    return cache[key]


__all__ = ["Cache", "eval_cache"]
