#!/usr/bin/env bash


gcloud compute ssh "$INSTANCE" \
       --command='sudo apt-get install python3-venv && '
                 'python3 -m venv venv && '
                 '. ~/venv/bin/activate && '
                 'pip3 install -U pip3 setuptools wheel &&
                 'pip3 install tensorflow && '
                 'curl -L https://github.com/tensorflow/models/archive/master.tar.gz -o models.tar.gz &&'
                 'tar xf models.tar.gz && '
                 'export PYTHONPATH="$HOME/models/official"'
