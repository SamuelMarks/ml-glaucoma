#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import pickle
import sys
from platform import python_version_tuple

import numpy as np
import tensorflow as tf
from PIL import Image

if python_version_tuple()[0] == '3':
    basestring = str


def _int64_feature(value):
    """Create a Int64List Feature

    :param value: The value to store in the feature
    :type  value: ``str``

    :return The FeatureEntry
    :rtype ``tf.train.Feature``
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Create a BytesList Feature

    :param value: The value to store in the feature
    :type  value: ``bytes``

    :return The FeatureEntry
    :rtype ``tf.train.Feature``
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(dataset_name, data_directory, class_map, segments=1, directories_as_labels=True,
                        pixels=224, files='**/*.jpg'):
    """Convert the dataset into TFRecords on disk
    
    :param dataset_name: The name/folder of the dataset
    :type  dataset_name: ``str``

    :param data_directory: The directory where records will be stored
    :type  data_directory: ``str``

    :param class_map: Dictionary mapping directory label name to integer label
    :type  class_map: ``dict``

    :param segments: The number of files on disk to separate records into
    :type  segments: ``str``

    :param segments: The number of files on disk to separate records into
    :type  segments: ``int``

    :param directories_as_labels: Whether the directory name should be used as it's label (used for test directory)
    :type  directories_as_labels: ``bool``

    :param files: Which files to find in the data directory
    :type  files: ``str`` or ``[str]``

    :param pixels: height and width value (one number)
    :type pixels: ``int`
    """

    # Create a dataset of file path and class tuples for each file
    do_glob = isinstance(files, basestring)

    filenames = glob.glob(os.path.join(data_directory, files)) if do_glob else (item
                                                                                for sublist in files
                                                                                for item in sublist)
    assert filenames, 'Nothing found matching \'{files}\' in: \'{data_directory}\''.format(
        files=files, data_directory=data_directory
    )
    classes = ((os.path.basename(os.path.dirname(name)) for name in filenames)
               if directories_as_labels else [None] * len(filenames))
    dataset = tuple(zip(filenames, classes))

    # If sharding the dataset, find how many records per file
    num_examples = len(filenames) if do_glob else len(files)
    samples_per_segment = num_examples // segments

    print('Have', samples_per_segment, 'per record file')

    for segment_index in range(segments):
        start_index = segment_index * samples_per_segment
        end_index = (segment_index + 1) * samples_per_segment

        sub_dataset = dataset[start_index:end_index]
        record_filename = os.path.join(data_directory, '{dataset_name}-{segment_index}.tfrecords'.format(
            dataset_name=dataset_name, segment_index=segment_index
        ))

        with tf.python_io.TFRecordWriter(record_filename) as writer:
            print('Writing', record_filename)

            for index, sample in enumerate(sub_dataset):
                sys.stdout.flush()

                file_path, label = sample
                image = Image.open(file_path)
                image = image.resize((pixels, pixels))
                image_raw = np.array(image).tostring()

                features = {
                    'label': _int64_feature(class_map[label]),
                    'text_label': _bytes_feature(label.encode()),
                    'image': _bytes_feature(image_raw)
                }
                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())


def process_directory(data_directory):
    """
    Process the directory to convert images to TFRecords

    :param data_directory: The data directory
    :type  data_directory: ``str``
    """

    data_dir = os.path.expanduser(data_directory)
    train_data_dir = os.path.join(data_dir, 'train')

    class_names = os.listdir(train_data_dir)  # Get names of classes
    class_name2id = {label: index for index, label in enumerate(class_names)}  # Map class names to integer labels

    # Persist this mapping so it can be loaded when training for decoding
    with open(os.path.join(data_directory, 'class_name2id.p'), 'wb') as p:
        pickle.dump(class_name2id, p, protocol=pickle.HIGHEST_PROTOCOL)

    convert_to_tfrecord('train', data_dir, class_name2id, segments=4)
    convert_to_tfrecord('validation', data_dir, class_name2id)
    convert_to_tfrecord('test', data_dir, class_name2id, directories_as_labels=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-directory',
        default=os.path.join(os.path.expanduser('~'), 'data', 'mnist'),
        help='Directory where TFRecords will be stored')

    args = parser.parse_args()
    process_directory(args.data_directory)
