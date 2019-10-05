#!/usr/bin/env python

from ml_glaucoma.cli_options import get_parser
from ml_glaucoma.utils import pp

if __name__ == '__main__':
    parser, commands = get_parser()
    kwargs = dict(parser.parse_args()._get_kwargs())
    command = kwargs.pop('command')
    if command is None:
        raise ReferenceError('You must specify a command. Append `--help` for details.')

    commands[command].build(**kwargs)
