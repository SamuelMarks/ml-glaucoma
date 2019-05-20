"""
Example usage:
python vis.py --gin_file='refuge_cls/dc0-base'
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from ml_glaucoma.flags import parse_config
from ml_glaucoma.problems import TfdsProblem


def main(_):
    import matplotlib.pyplot as plt
    parse_config()
    problem = TfdsProblem()
    dataset = problem.get_dataset('train')
    for args in dataset:
        inputs, label = args[:2]
        if inputs.shape[-1] == 1:
            inputs = tf.squeeze(inputs, axis=-1)
        inputs -= tf.reduce_min(inputs)
        inputs /= tf.reduce_max(inputs)
        plt.imshow(inputs.numpy())
        plt.title('Glaucoma' if label.numpy() else 'Non-Glaucoma')
        plt.show()


if __name__ == '__main__':
    app.run(main)
