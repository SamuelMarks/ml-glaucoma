"""
Example usage:
python train.py --gin_file='refuge_cls/dc0-base'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from ml_glaucoma import flags


def main(_):
    from ml_glaucoma import runners
    flags.parse_config()
    runners.train()


if __name__ == '__main__':
    app.run(main)
