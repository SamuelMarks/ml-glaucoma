from collections import deque
from datetime import datetime
from itertools import islice
from os import environ
from pprint import PrettyPrinter

redis_cursor = None
if 'NO_REDIS' not in environ:
    from redis import StrictRedis

    redis_cursor = StrictRedis(host='localhost', port=6379, db=0)

pp = PrettyPrinter(indent=4).pprint

it_consumes = lambda it, n=None: deque(it, maxlen=0) if n is None else next(islice(it, n, n), None)


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
    raise TypeError ("Type not serializable")
