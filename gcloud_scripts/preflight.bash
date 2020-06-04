#!/usr/bin/env bash

if [ -z "$NETWORK" ]; then
    printf '$NETWORK not defined; try including all environment variables specified in README.md\n' >&2
    exit 2
fi
