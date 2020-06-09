from abc import abstractmethod


def _examples_per_epoch(builder, split):
    return int(builder.info.splits[split].num_examples)


class Problem(object):
    """
    Abstract base class representing  data and evaluation.

    See `BaseProblem` for base implementation.
    """

    @abstractmethod
    def get_dataset(self, split, repeat=False, batch_size=None):
        raise NotImplementedError

    @abstractmethod
    def examples_per_epoch(self, split=None):  # `split=tfds.Split.TRAIN`
        raise NotImplementedError

    @abstractmethod
    def loss(self):
        raise NotImplementedError

    # -------------------------------------------------
    # Base implementations: these make possibly wrong assumptions
    @property
    def metrics(self):
        return None

    def output_spec(self):
        """Assumed to be the same as target_spec, but not necessarily."""
        return self.target_spec()

    # -------------------------------------------------
    # Base implementations: these are possibly inefficient
    def dataset_spec(self):
        """Structure of `tf.keras.layers.InputSpec` corresponding to output."""
        raise NotImplementedError()

    def input_spec(self):
        """First output (index 0) of `dataset_spec`."""
        return self.dataset_spec()[0]

    def target_spec(self):
        """Second output (index 1) of `dataset_spec`."""
        return self.dataset_spec()[1]


del abstractmethod

__all__ = ['Problem', '_examples_per_epoch']
