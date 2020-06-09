import os
from contextlib import suppress
from sys import modules

import tensorflow as tf

from ml_glaucoma import callbacks as cb, runners, get_logger
from ml_glaucoma.cli_options.logparser.utils import parse_line
from ml_glaucoma.runners.utils import default_model_dir, batch_steps

logger = get_logger(modules[__name__].__name__)


# logging.getLogger('dataset_builder').setLevel(logging.WARNING)


def train(problem, batch_size, epochs,
          model_fn, optimizer, class_weight=None,
          model_dir=None, callbacks=None, verbose=True,
          checkpoint_freq=5, summary_freq=10, lr_schedule=None,
          tensorboard_log_dir=None, write_images=False,
          cv=10, tpu=None):
    """
    Train a model on the given problem

    :param problem: Problem instance
    :param problem: ```ml_glaucoma.problems.Problem```

    :param batch_size: size of each batch for training/evaluation
    :param batch_size: int

    :param epochs: number of epochs
    :param epochs: int

    :param model_fn: function mapping (inputs, output_spec) -> outputs.
    :param model_fn: (inputs, output_spec) -> outputs

    :param optimizer: Optimizer instance.
    :param optimizer: ```tf.keras.optimizers.Optimizer```

    :param class_weight: Optional dictionary mapping class indices (integers)
            to a weight (float) value, used for weighting the loss function
            (during training only).
            This can be useful to tell the model to
            "pay more attention" to samples from
            an under-represented class.
    :param class_weight: {}

    :param model_dir: string path to directory containing weight files.
    :param model_dir: str

    :param callbacks: list of callbacks in addition to those created below
    :param callbacks: [tf.keras.callbacks.Callback]

    :param verbose: passed to `tf.keras.models.Model.fit`
    :param verbose: bool

    :param checkpoint_freq: frequency in epochs at which to save weights.
    :param checkpoint_freq: int

    :param summary_freq: frequency in batches at which to save tensorboard summaries.
    :param summary_freq: int

    :param lr_schedule: function mapping `epoch -> learning_rate`
    :param (int) -> int

    :param tensorboard_log_dir: directory to log tensorboard summaries. If not
            provided, `model_dir` is used
    :param tensorboard_log_dir: str

    :param write_images: passed to `tf.keras.callbacks.TensorBoard`
    :param write_images: bool

    :param write_images: passed to `tf.keras.callbacks.TensorBoard`
    :param write_images: bool

    :param cv: cross-validation amount
    :param cv: int

    :param tpu: TPU to use
    :param tpu: None or URL of TPU

    :return `History` object as returned by `model.fit`
    :rtype ``tf.keras.History``
    """
    optimizer = optimizer  # type: tf.keras.optimizers.Optimizer
    if model_dir is None:
        model_dir = default_model_dir()
    model_dir = os.path.expanduser(model_dir)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if tpu is not None:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu if tpu.startswith('grpc') else 'grpc://{}'.format(tpu)
        )
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
        strategy_scope = strategy.scope
    else:
        strategy_scope = suppress

    with strategy_scope():
        train_ds, val_ds, test_ds = tf.nest.map_structure(
            lambda split: problem.get_dataset(split, batch_size, repeat=True),
            ('train', 'validation', 'test')
        )
        val_ds = val_ds.concatenate(test_ds)

        # new_splits = train_ds + val_ds
        # new_val = new_splits[0,15,1,4,3,6]
        # new_train = new_splits[1,16,2,5,4,7]
        # Your job is to generate these indexes, such that for 10 sets of val, train indexes, they are sufficiently different (e.g.: 20% variance)?

        # Create 10 selections of train / validation data

        inputs = tf.nest.map_structure(
            lambda spec: tf.keras.layers.Input(
                shape=spec.shape, dtype=spec.dtype),
            problem.input_spec())
        model = model_fn(inputs, problem.output_spec())  # type: tf.keras.Model
        model.compile(
            optimizer=optimizer,
            loss=problem.loss,
            metrics=problem.metrics)

    train_steps = batch_steps(
        problem.examples_per_epoch('train'), batch_size)
    validation_steps = batch_steps(
        problem.examples_per_epoch('validation'), batch_size)

    common_callbacks, initial_epoch = cb.backends.get_callbacks(
        batch_size=batch_size,
        checkpoint_freq=checkpoint_freq,
        summary_freq=summary_freq,
        model_dir=model_dir,
        train_steps_per_epoch=train_steps,
        val_steps_per_epoch=validation_steps,
        lr_schedule=lr_schedule,
        tensorboard_log_dir=tensorboard_log_dir,
        write_images=write_images,
    )
    if callbacks is None:
        callbacks = common_callbacks
    else:
        callbacks.extend(common_callbacks)

    try:
        dot = tf.keras.utils.model_to_dot(model)
        if dot is not None:
            dotfile = os.path.join(os.path.dirname(model_dir),
                                   os.path.basename(model_dir) + '.dot')
            dot.write(dotfile)
            print('graphviz diagram of model generated to:', dotfile)
    except ImportError:
        logger.warn('Install graphviz and pydot to generate graph')

    if model.name == 'model':
        model._name = os.path.basename(model_dir)

    model.summary()

    parsed_line = parse_line(os.path.basename(model_dir))

    just = 15
    print(
        'dataset:'.ljust(just), parsed_line.dataset, '\n',
        'transfer:'.ljust(just), parsed_line.transfer, '\n',
        'optimizer:'.ljust(just), optimizer.__class__.__name__, '\n',
        'loss:'.ljust(just), problem.loss.__class__.__name__, '\n',
        'callbacks:'.ljust(just), ', '.join(map(lambda c: type(c).__name__, callbacks)), '\n',
        'metrics:'.ljust(just), ', '.join(map(lambda m: type(m).__name__, problem.metrics)), '\n',
        'total_epochs:'.ljust(just), epochs, '\n',
        '_' * 65, '\n',
        sep=''
    )

    # Fit model against all 10 selections

    fit_result = model.fit(
        train_ds,
        epochs=epochs,
        class_weight=class_weight,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=val_ds,
        steps_per_epoch=train_steps,
        validation_steps=validation_steps,
        initial_epoch=initial_epoch,
    )

    return fit_result


def evaluate(problem, batch_size, model_fn, optimizer, model_dir=None):
    """
    Evaluate the given model with weights saved as `model_dir`.

    Args:
        problem: `ml_glaucoma.problems.Problem` instance
        batch_size: size of each batch
        model_fn: model_fn: function mapping (inputs, output_spec) -> outputs
        optimizer: `tf.keras.optimizers.Optimizer` instance
        model_dir: string path to directory containing weight files.

    Returns:
        scalar or list of scalars - loss/metrics values
        (output of `tf.keras.models.Model.evaluate`)
    """
    if model_dir is None:
        model_dir = default_model_dir()
    if model_dir is not None:
        model_dir = os.path.expanduser(model_dir)
    if not os.path.isdir(model_dir):
        raise RuntimeError('model_dir does not exist: %s' % model_dir)

    val_ds = problem.get_dataset('validation', batch_size, repeat=False)
    inputs = tf.nest.map_structure(
        lambda spec: tf.keras.layers.Input(
            shape=spec.shape, dtype=spec.dtype),
        problem.input_spec())
    model = model_fn(inputs, problem.output_spec())
    model.compile(
        optimizer=optimizer,
        loss=problem.loss,
        metrics=problem.metrics)

    manager_cb = cb.LoadingModelCheckpoint(
        model_dir=model_dir, save_freq='epoch'
    )
    manager_cb.set_model(model)
    manager_cb.restore()

    validation_steps = batch_steps(
        problem.examples_per_epoch('validation'), batch_size)

    return runners.evaluate(val_ds, steps=validation_steps)


def vis(problem, split='train'):
    """
    Simple visualization of the given `ml_glaucoma.problems.Problem` instance.

    Requires `matplotlib`.
    """
    import matplotlib.pyplot as plt
    tf.compat.v1.enable_eager_execution()
    for fundus, label in problem.get_dataset(split=split):
        if fundus.shape[-1] == 1:
            fundus = tf.squeeze(fundus, axis=-1)
        fundus -= tf.reduce_min(input_tensor=fundus, axis=(0, 1))
        fundus /= tf.reduce_max(input_tensor=fundus, axis=(0, 1))
        plt.imshow(fundus.numpy())
        plt.title('Glaucoma' if label.numpy() else 'Non-glaucoma')
        plt.show()


# Cleanup namespace
# del modules, cb, runners, get_logger

__all__ = ['train', 'evaluate', 'vis']
