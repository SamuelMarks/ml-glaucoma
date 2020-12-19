from os import environ

import tensorflow as tf

if __name__ == "__main__":
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu="grpc://{}".format(environ["COLAB_TPU_ADDR"])
    )
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices("TPU"))
