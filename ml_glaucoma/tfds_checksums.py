"""
Hacky work-around to https://github.com/tensorflow/datasets/issues/580

Resolved in master branch, though required for earlier versions.

Note there's a @memoize on `tfds.core.download.checksums._checksum_paths()`,
so this should be imported before any other `tfds` usage.
"""

from __future__ import division, print_function

from os import environ

if not environ["TF"]:
    raise NotImplementedError("tfds_checksums is TensorFlow only")

try:
    import os

    from tensorflow_datasets.core import download

    download.checksums.add_checksums_dir(
        os.path.realpath(os.path.join(os.path.dirname(__file__), "url_checksums"))
    )
    download.checksums._checksum_paths.cache_clear()
except AttributeError:
    # later versions of tfds don't have tfds.core.download.checksums
    # bug seems fixed in these?
    pass
