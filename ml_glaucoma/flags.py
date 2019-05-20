from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging
import gin
import os

flags.DEFINE_string(
  'config_dir', '__HERE__/../config', 'root config directory to change to')
flags.DEFINE_string(
  'gin_file', None, 'List of paths to the config files.')
flags.DEFINE_multi_string(
  'gin_param', None, 'Newline separated list of Gin parameter bindings.')

FLAGS = flags.FLAGS


def parse_config():
    import ml_glaucoma.runners
    gin_file = FLAGS.gin_file
    if not gin_file.endswith('.gin'):
        gin_file = '%s.gin' % gin_file
    config_dir = os.path.realpath(os.path.expanduser(FLAGS.config_dir.replace(
      '__HERE__', os.path.dirname(__file__))))
    logging.info(
      'Searching for config at %s' % os.path.join(config_dir, gin_file))
    os.chdir(config_dir)
    gin.bind_parameter('default_model_dir.model_id', gin_file[:-4])
    gin.parse_config_files_and_bindings([gin_file], FLAGS.gin_param)
    logging.info('Config loaded successfully loaded.')
