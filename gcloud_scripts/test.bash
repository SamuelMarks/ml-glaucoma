#!/usr/bin/env bash

set -euo pipefail

gcloud compute ssh "$INSTANCE" \
       --command='( dpkg -s python3-venv &> /dev/null || sudo apt-get install -y python3-venv ) &&
                  ( [ -d venv ] || python3 -m venv venv ) &&
                  . ~/venv/bin/activate &&
                  ( python -c "import pkgutil; exit(int(pkgutil.find_loader(\"tensorflow\") is not None))" &&
                    pip3 install -U pip setuptools wheel &&
                    pip3 install tensorflow ) &&
                  ( [ -d models-master ] || ( curl -L https://github.com/tensorflow/models/archive/master.tar.gz -o models.tar.gz &&
                    tar xf models.tar.gz ) ) &&
                  export PYTHONPATH='~/models/official''
