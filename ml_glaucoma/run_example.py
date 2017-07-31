from argparse import Namespace
from subprocess import call

import tensorflow as tf
from os import path

import ml_glaucoma.examples.image_retraining.retrain

# call('echo hello world')
if not path.isdir('/tf_files'):
    call('mkdir /tf_files')
if not path.isfile('/tf_files/flower_photos.tgz'):
    call('curl http://download.tensorflow.org/example_images/flower_photos.tgz -o /tf_files/flower_photos.tgz')
    call('cd /tf_files && tar xzf flower_photos.tgz')
if not path.isdir('/tensorflow'):
    call('git clone https://github.com/tensorflow/tensorflow.git /tensorflow')

ml_glaucoma.examples.image_retraining.retrain.FLAGS = Namespace(
    bottleneck_dir='/tf_files/bottlenecks', eval_step_interval=10, final_tensor_name='final_result',
    flip_left_right=False, how_many_training_steps=15000, image_dir='/tf_files/flower_photos',
    learning_rate=0.01, model_dir='/tf_files/inception', output_graph='/tf_files/retrained_graph.pb',
    output_labels='/tf_files/retrained_labels.txt', print_misclassified_test_images=False,
    random_brightness=0, random_crop=0, random_scale=0, summaries_dir='/tmp/retrain_logs',
    test_batch_size=-1, testing_percentage=10, train_batch_size=100, validation_batch_size=100,
    validation_percentage=10)
unparsed = []
tf.app.run(main=ml_glaucoma.examples.image_retraining.retrain.main,
           argv=['tensorflow/examples/image_retraining/retrain.py'])
'''
2017-03-23 08:20:54.546703: I tensorflow/compiler/xla/service/platform_util.cc:58] platform Host present with 8 visible devices
2017-03-23 08:20:54.549206: I tensorflow/compiler/xla/service/service.cc:183] XLA service 0x29a7c60 executing computations on platform Host. Devices:
2017-03-23 08:20:54.549244: I tensorflow/compiler/xla/service/service.cc:191]   StreamExecutor device (0): <undefined>, <undefined>
2017-03-23 08:20:54.706135: Step 0: Train accuracy = 33.0%
...
2017-03-23 08:44:05.897891: Step 14980: Train accuracy = 99.0%
2017-03-23 08:44:05.897983: Step 14980: Cross entropy = 0.058768
2017-03-23 08:44:05.980637: Step 14980: Validation accuracy = 94.0% (N=100)
2017-03-23 08:44:06.828600: Step 14990: Train accuracy = 100.0%
2017-03-23 08:44:06.828706: Step 14990: Cross entropy = 0.056074
2017-03-23 08:44:06.913495: Step 14990: Validation accuracy = 90.0% (N=100)
2017-03-23 08:44:07.678602: Step 14999: Train accuracy = 98.0%
2017-03-23 08:44:07.678694: Step 14999: Cross entropy = 0.069492
2017-03-23 08:44:07.760873: Step 14999: Validation accuracy = 93.0% (N=100)
Final test accuracy = 91.7% (N=387)
'''
