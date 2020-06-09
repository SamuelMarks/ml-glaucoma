import os


def batch_steps(num_examples, batch_size):
    """Get the number of batches, including possible fractional last."""
    steps = num_examples // batch_size
    if num_examples % batch_size > 0:
        steps += 1
    return steps


def default_model_dir(base_dir=os.path.join(os.path.expanduser('~'), 'ml_glaucoma_models'), model_id=None):
    """
    Get a new directory at `base_dir/model_id`.

    If model_id is None, we use 'model{:03d}', counting up from 0 until we find
    a space, i.e. model000, model001, model002 ...
    """
    if model_id is None:
        i = 0
        model_dir = os.path.join(base_dir, 'model{:03d}'.format(i))
        while os.path.isdir(model_dir):
            i += 1
            model_dir = os.path.join(base_dir, 'model{:03d}'.format(i))
    else:
        model_dir = os.path.join(base_dir, model_id)
    return model_dir


__all__ = ['batch_steps', 'default_model_dir']
