import os
from stat import S_IWOTH, S_IWGRP, S_IWRITE
from sys import modules

import tensorflow as tf

from ml_glaucoma import callbacks as cb, runners, get_logger
from ml_glaucoma.cli_options.logparser.tf import log_parser
from ml_glaucoma.cli_options.logparser.utils import parse_line
from ml_glaucoma.runners.utils import default_model_dir, batch_steps

logger = get_logger(modules[__name__].__name__)


# logging.getLogger('dataset_builder').setLevel(logging.WARNING)


def train(problem, batch_size, epochs,
          model_fn, optimizer, class_weight=None,
          model_dir=None, callbacks=None, verbose=True,
          checkpoint_freq=5, summary_freq=10, lr_schedule=None,
          tensorboard_log_dir=None, write_images=False, delete_lt=None):
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

    :param delete_lt: delete *.h5 files that are less than this threshold
    :param delete_lt: float

    :return `History` object as returned by `model.fit`
    :rtype ``tf.keras.History``
    """
    optimizer = optimizer  # type: tf.keras.optimizers.Optimizer
    if model_dir is None:
        model_dir = default_model_dir()
    model_dir = os.path.expanduser(model_dir)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    train_ds, val_ds = tf.nest.map_structure(
        lambda split: problem.get_dataset(split, batch_size, repeat=True),
        ('train', 'validation'))
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
        '_' * 98, '\n',
        sep=''
    )

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

    if delete_lt is not None:
        result = log_parser(directory=os.path.join(callbacks[-1].log_dir, 'validation'), top=1,
                            tag='epoch_auc', infile=None, by_diff=None, threshold=None, rest=None)
        if result is None:
            return None
        else:
            dire, best_runs = result
            print('{} ({}) had a best_runs of {}'.format(dire, callbacks[-1].log_dir, best_runs))
            #  if not next((True for run in best_runs if run < delete_lt), False):
            if best_runs[0][1] < delete_lt:
                print('Insufficient AUC ({}) for storage, '
                      'removing h5 files to save disk space. `dire`:'.format(best_runs[0][1]), dire)
                if os.path.isfile(dire):
                    dire = os.path.dirname(dire)
                root = os.path.splitdrive(os.getcwd())[0] or '/'
                while not os.path.isfile(os.path.join(dire, 'model-0001.h5')):
                    dire = os.path.dirname(dire)
                    if dire == root:
                        raise EnvironmentError('No h5 files generated')

                for fname in os.listdir(dire):
                    full_path = os.path.join(dire, fname)
                    if os.path.isfile(full_path) and full_path.endswith('h5'):
                        os.remove(full_path)
                        if os.path.isfile(full_path):
                            from pathlib import Path

                            Path(full_path).unlink()

                # Make directory read-only
                mode = os.stat(dire).st_mode
                ro_mask = 0o777 ^ (S_IWRITE | S_IWGRP | S_IWOTH)
                os.chmod(dire, mode & ro_mask)
            else:
                print('{} >= {}; so not removing h5 files'.format(best_runs[0][1], delete_lt))
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
