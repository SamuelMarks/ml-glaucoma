from os import path
from platform import python_version_tuple
from random import sample
from string import ascii_uppercase

import quantumrandom

if python_version_tuple()[0] == "3":
    from functools import reduce

    imap = map
else:
    from itertools import imap


def get_upper_kv(module):
    return {k: getattr(module, k) for k in dir(module) if k[0] in ascii_uppercase}


def create_random_numbers(minimum, maximum, n):  # type: (int, int, int) -> [int]
    whole, prev = frozenset(), frozenset()
    while len(whole) < n:
        whole = reduce(
            frozenset.union,
            (
                frozenset(
                    imap(
                        lambda num: minimum + (num % maximum),
                        quantumrandom.get_data(data_type="uint16", array_length=1024),
                    )
                ),
                prev,
            ),
        )
        prev = whole
        print(len(whole), "of", n)
    return sample(whole, n)


def ensure_is_dir(filepath):  # type: (str) -> str
    assert filepath is not None and path.isdir(
        filepath
    ), "{!r} is not a directory".format(filepath)
    return filepath


__all__ = ["get_upper_kv", "create_random_numbers", "ensure_is_dir"]
