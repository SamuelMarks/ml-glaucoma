#!/usr/bin/env bash

declare -r DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -euo pipefail

"$DIR/preflight.bash"

declare -r TPU_ADDR="$(gcloud compute tpus describe $TPU_NAME --format='value[separator=":"](networkEndpoints.ipAddress, networkEndpoints.port)')"

gcloud compute ssh "$INSTANCE" \
    --command='( dpkg -s python3-venv &> /dev/null || sudo apt-get update -qq && sudo apt-get install -y python3-venv ) &&
                  ( [ -d venv ] || python3 -m venv venv ) &&
                  . ~/venv/bin/activate &&
                  ( python -c "import pkgutil; exit(int(pkgutil.find_loader(\"tensorflow\") is not None))" &&
                    pip3 install -U pip setuptools wheel &&
                    pip3 install tensorflow tensorflow-datasets nbconvert ||
                    true ) &&
                  ( [ -f tpu-tester.py ] ||
                    curl -sL https://raw.githubusercontent.com/tensorflow/docs/945a448/site/en/guide/tpu.ipynb |
                    jupyter nbconvert --to script --stdin --output tpu-tester.py &&
                    sed "s/%/pass \#/g" tpu-tester.py.txt > tpu-tester.py ) &&
                  COLAB_TPU_ADDR="'"$TPU_ADDR"'" python tpu-tester.py'
