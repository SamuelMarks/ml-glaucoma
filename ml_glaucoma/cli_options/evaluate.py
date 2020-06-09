import ml_glaucoma.runners
from ml_glaucoma.cli_options.base import Configurable


class ConfigurableEvaluate(Configurable):
    description = 'Evaluate model'

    def __init__(self, problem, model_fn, optimizer):
        super(ConfigurableEvaluate, self).__init__(
            problem=problem, model_fn=model_fn, optimizer=optimizer)

    def fill_self(self, parser):
        parser.add_argument(
            '-b', '--batch_size', default=32, type=int,
            help='size of each batch')
        parser.add_argument(
            '--model_dir',
            help='model directory in which to save weights and tensorboard '
                 'summaries')

    def build_self(self, problem, batch_size, model_fn, optimizer, model_dir,
                   **kwargs):
        return ml_glaucoma.runners.evaluate(
            problem=problem,
            batch_size=batch_size,
            model_fn=model_fn,
            optimizer=optimizer,
            model_dir=model_dir,
        )


__all__ = ['ConfigurableEvaluate']
