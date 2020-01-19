#!/usr/bin/env bash

declare -r DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

set -euov pipefail

declare -r TPU_ADDR="$(gcloud compute tpus describe $TPU_NAME --format='value[separator=":"](networkEndpoints.ipAddress, networkEndpoints.port)')"

gcloud compute scp --zone "$ZONE" --project="$PROJECT_ID" \
                   "$DIR"'/tpu-tester.py'  "$INSTANCE":~/

printf '$TPU_ADDR = "%s"' "$TPU_ADDR"

gcloud compute ssh "$INSTANCE" \
       --command='( dpkg -s python3-venv &> /dev/null || sudo apt-get install -y python3-venv ) &&
                  ( [ -d venv ] || python3 -m venv venv ) &&
                  . ~/venv/bin/activate &&
                  ( python -c "import pkgutil; exit(int(pkgutil.find_loader(\"tensorflow\") is not None))" &&
                    pip3 install -U pip setuptools wheel &&
                    pip3 install tensorflow tensorflow-datasets ||
                    true ) &&
                  COLAB_TPU_ADDR="'"$TPU_ADDR"'" python tpu-tester.py'
