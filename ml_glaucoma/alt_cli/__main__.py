from ml_glaucoma.alt_cli import get_parser

if __name__ == '__main__':
    main_parser, commands = get_parser()
    kwargs = dict(main_parser.parse_args()._get_kwargs())
    command = kwargs.pop('command')
    if command is None:
        raise ReferenceError(
            'You must specify a command. Append `--help` for details.')

    commands[command].build(**kwargs)
