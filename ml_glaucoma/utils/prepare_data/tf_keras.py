import tensorflow as tf

import ml_glaucoma.runners.tf_keras


def prepare_data(data_obj, pixels):
    def _parse_function(filename, label):
        image = tf.image.decode_jpeg(tf.io.read_file(filename), channels=3)
        print(image.shape)
        image_resized = tf.image.resize(image, [pixels, pixels])
        print(image_resized.shape)
        return image_resized, label

    def _get_filenames(dataset, pos_ids, id_to_imgs):
        # returns filenames list and labels list
        labels = []
        filenames = []
        for curr_id in dataset:
            for filename in id_to_imgs[curr_id]:
                if curr_id in pos_ids:
                    labels += [1]
                else:
                    labels += [0]
                filenames += [filename]
        labels = tf.constant(labels)
        filenames = tf.constant(filenames)
        return filenames, labels

    def _create_dataset(data, dataset):
        pos_ids = data.pickled_cache["oags1"]
        id_to_imgs = data.pickled_cache["id_to_imgs"]

        img_names, data_labels = _get_filenames(dataset, pos_ids, id_to_imgs)
        dataset = tf.data.Dataset.from_tensor_slices((img_names, data_labels))
        dataset = dataset.map(_parse_function)
        # dataset = tf.map_fn(_parse_function,dataset)
        return dataset

    train = _create_dataset(data_obj, ml_glaucoma.runners.tf_keras.train)
    validation = _create_dataset(data_obj, data_obj.datasets.validation)
    test = _create_dataset(data_obj, data_obj.datasets.test)
    return train, validation, test


del tf

__all__ = ["prepare_data"]
