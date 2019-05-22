"""
Hacky work-around to https://github.com/tensorflow/datasets/issues/580

Note there's a @memoize on `tfds.core.download.checksums._checksum_paths()`,
so this should be imported before any other `tfds` usage.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow_datasets as tfds


tfds.core.download.checksums._CHECKSUM_DIRS.append(
    os.path.realpath(os.path.join(os.path.dirname(__file__), 'url_checksums')))
