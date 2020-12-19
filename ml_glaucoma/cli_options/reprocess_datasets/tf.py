import tensorflow as tf
from tensorflow.core.util import event_pb2


def reprocess_datasets(filepaths):
    print("process_dataset::filepaths:", filepaths)
    for filepath in filepaths:
        for record in tf.data.TFRecordDataset(filepath):
            event = event_pb2.Event.FromString(tf.get_static_value(record))
            if event.HasField("summary"):
                pass  # We don't care about summaries
            else:
                if reprocess_datasets.t > 0:
                    reprocess_datasets.t -= 1
                    print("dir(event):", dir(event))
                    print("dir(record):", dir(record))
                    print("event.ListFields():", event.ListFields())


reprocess_datasets.t = 2

__all__ = ["reprocess_datasets"]
