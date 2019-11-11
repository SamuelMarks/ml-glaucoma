import tensorflow as tf

from ml_glaucoma import callbacks as callbacks_module
from ml_glaucoma.utils.helpers import get_upper_kv

valid_callbacks = get_upper_kv(tf.keras.callbacks)
valid_callbacks.update(get_upper_kv(callbacks_module))
SUPPORTED_CALLBACKS = tuple(sorted(valid_callbacks.keys()))

# Cleanup namespace
del callbacks_module
del tf
del get_upper_kv
