#!/usr/bin/env python
from sys import argv

from ml_glaucoma.cli_options.parser import cli_handler

if __name__ == "__main__":
    print("sys.argv", argv)
    cli_handler()
