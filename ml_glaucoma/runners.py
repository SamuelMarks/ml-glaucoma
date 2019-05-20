"""Train/validation/predict loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from absl import logging
import gin
from ml_glaucoma import callbacks as cb


def batch_steps(num_examples, batch_size):
    steps = num_examples // batch_size
    if num_examples % batch_size > 0:
        steps += 1
    return steps


@gin.configurable
def default_model_dir(base_dir='~/ml_glaucoma_models', model_id=None):
    if model_id is None:
        i = 0
        model_dir = os.path.join(base_dir, 'model%03d' % i)
        while os.path.isdir(model_dir):
            i += 1
            model_dir = os.path.join(base_dir, 'model%03d' % i)
    else:
        model_dir = os.path.join(base_dir, model_id)
    return model_dir


@gin.configurable
def train(
        problem, batch_size, epochs, model_fn, optimizer, model_dir=None,
        callbacks=None, verbose=True, checkpoint_freq=5,
        summary_freq=10, save_config=True, lr_schedule=None):
    if model_dir is None:
        model_dir = default_model_dir()
    if model_dir is not None:
        model_dir = os.path.expanduser(model_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    train_ds, val_ds = tf.nest.map_structure(
        lambda split: problem.get_dataset(split, batch_size, repeat=True),
        ('train', 'validation'))
    inputs = tf.nest.map_structure(
        lambda spec: tf.keras.layers.Input(shape=spec.shape, dtype=spec.dtype),
        problem.input_spec())
    model = model_fn(inputs, problem.output_spec())
    model.compile(
        optimizer=optimizer,
        loss=problem.loss,
        metrics=problem.metrics)

    train_steps = batch_steps(problem.examples_per_epoch('train'), batch_size)
    validation_steps = batch_steps(
        problem.examples_per_epoch('validation'), batch_size)

    callbacks, initial_epoch = cb.get_callbacks(
        model,
        batch_size=batch_size,
        callbacks=callbacks,
        checkpoint_freq=checkpoint_freq,
        summary_freq=summary_freq,
        save_config=save_config,
        model_dir=model_dir,
        train_steps_per_epoch=train_steps,
        val_steps_per_epoch=validation_steps,
        lr_schedule=lr_schedule,
    )

    model.fit(
        train_ds,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=val_ds,
        steps_per_epoch=train_steps,
        validation_steps=validation_steps,
        initial_epoch=initial_epoch,
    )


@gin.configurable
def evaluate(problem, batch_size, model_fn, optimizer, model_dir=None):
    if model_dir is None:
        model_dir = default_model_dir()
    if model_dir is not None:
        model_dir = os.path.expanduser(model_dir)
    if not os.path.isdir(model_dir):
        raise RuntimeError('model_dir does not exist: %s' % model_dir)
    val_ds = problem.get_dataset('validation', batch_size, repeat=False)
    inputs = tf.nest.map_structure(
        lambda spec: tf.keras.layers.Input(shape=spec.shape, dtype=spec.dtype),
        problem.input_spec())
    model = model_fn(inputs, problem.output_spec())
    model.compile(
        optimizer=optimizer,
        loss=problem.loss,
        metrics=problem.metrics)

    checkpoint = tf.train.Checkpoint(model=model)
    manager_cb = cb.CheckpointManagerCallback(
        model_dir=model_dir, period=1, max_to_keep=5)
    manager_cb.set_model(model)
    manager_cb.restore()

    validation_steps = batch_steps(
        problem.examples_per_epoch('validation'), batch_size)

    model.evaluate(val_ds, steps=validation_steps)
