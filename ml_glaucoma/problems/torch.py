from abc import abstractmethod
from sys import modules

from ml_glaucoma import get_logger
from ml_glaucoma.problems.shared import Problem
from ml_glaucoma.utils import pp

logger = get_logger(modules[__name__].__name__)


def dataset_spec_to_input_spec(*args, **kwargs):
    raise NotImplementedError()


def preprocess_example(*args, **kwargs):
    if 'image' not in kwargs:
        logger.warn('Image not in kwargs; skipping preprocess_example')
        return
    print('preprocess_example')
    pp(kwargs)
    raise NotImplementedError()


class BaseProblem(Problem):
    """
    Almost-complete basic implementation of `Problem`.

    Requires the following methods to be implemented:
        * _get_base_dataset
        * examples_per_epoch
    """

    def __init__(self, loss, metrics, shuffle_buffer=1024,
                 map_fn=None, use_inverse_freq_weights=False,
                 class_counts=None, output_spec=None,
                 dataset_spec=None):
        """
        Base implementation of `Problem`.

        Args:
            loss: `tf.keras.losses.Loss` instance
            metrics: iterable of `tf.keras.metrics.Metric` instances
            shuffle_buffer: used in `tf.data.Dataset.shuffle`
            map_fn: optional map function/dict of split -> map_fn that is
                applied to each example independently.
                Used in `tf.data.Dataset.map` prior to batching.
            use_inverse_freq_weights: for classification problems, this creates
                weights based on inverse class frequencies.
            class_counts (optional): iterable of ints, class counts. If given
                and `use_inverse_freq_weights`, the these are the frequencies
                used. There is no requirement that they correspond to the
                actual class counts, but the weights will be calculated as if
                they are. If not provided, the dataset is reduced to count the
                relevant weights.
            output_spec: (optional) structure of `tf.keras.layers.InputSpec`s
                corresponding to the desired output. If `None`, this is assumed
                to be the same as `target_spec`, i.e. the spec of the second
                output of `self.dataset_spec()`, though this may be incorrect
                (e.g. in sparse classification problems, where outputs (logits)
                should have 1 extra dimension compared to the sparse labels and
                be floats).
            dataset_spec: (optional) structure of `tf.keras.layers.InputSpec`s
                corresponding to the output of `self.get_dataset()`. Providing
                this explicitly removes the necessity to build the dataset in
                order to calculate it.
        """
        self._map_fn = map_fn
        self._use_inverse_freq_weights = use_inverse_freq_weights
        self._class_counts = class_counts
        if shuffle_buffer is None:
            shuffle_buffer = self.examples_per_epoch('train')
        self._shuffle_buffer = shuffle_buffer
        self._loss = loss
        self._metrics = tuple(metrics)
        self._output_spec = output_spec
        self._dataset_spec = dataset_spec

    def output_spec(self):
        """Assumed to be the same as target_spec, but not necessarily."""
        raise NotImplementedError()

    def dataset_spec(self):
        raise NotImplementedError()

    @property
    def loss(self):
        raise NotImplementedError()

    @property
    def metrics(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_base_dataset(self, split):
        raise NotImplementedError()

    def get_dataset(self, split, batch_size=None, repeat=False, prefetch=True):
        raise NotImplementedError()

    def data_pipeline(
        self, dataset, split, batch_size, repeat, shuffle=None,
        prefetch=True):
        raise NotImplementedError()

    @property
    def class_counts(self):
        raise NotImplementedError()


def TfdsProblem(*args, **kwargs):
    raise NotImplementedError()


class TfdsMultiProblem(BaseProblem):
    """Problem based on multiple `tfds.DatasetBuilder`s."""

    def __init__(self, builders, *args, **kwargs):
        """
        Args:
            builders: iterable of `tfds.core.DatasetBuilder`s
            args/kwargs: see `ml_glaucoma.problems.BaseProblem`.
        """
        self._builders = tuple(builders)
        super(TfdsMultiProblem, self).__init__(*args, **kwargs)

    def get_dataset(self, split, batch_size=None, repeat=False, prefetch=True):
        raise NotImplementedError()

    def examples_per_epoch(self, split=None):
        raise NotImplementedError()

    def input_spec(self):
        raise NotImplementedError()

    def target_spec(self):
        """`target` means label."""
        raise NotImplementedError()


def OLD____TfdsMultiProblem(*args, **kwargs):
    print('---' * 10)
    print('TfdsMultiProblem')
    pp(args)
    pp(kwargs)

    print('---' * 10)
    print('map_fn')
    pp(kwargs['map_fn']['test']())
    print('---' * 10)

    '''train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
    test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    class CNN(nn.Module):
        pass

    # omitted...

    if __name__ == '__main__':
        print("Number of train samples: ", len(train_data))
        print("Number of test samples: ", len(test_data))
        print("Detected Classes are: ", train_data.class_to_idx)  # classes are detected by folder structure

        model = CNN()'''

    # raise NotImplementedError()


def with_inverse_freq_weights(*args, **kwargs):
    raise NotImplementedError()


del abstractmethod, modules, get_logger, Problem, pp

__all__ = ['dataset_spec_to_input_spec', 'preprocess_example', 'BaseProblem',
           'TfdsProblem', 'TfdsMultiProblem', 'with_inverse_freq_weights']
