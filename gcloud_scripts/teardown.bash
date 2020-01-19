#!/usr/bin/env bash

declare -r DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

set -euo pipefail

"$DIR/preflight.bash"

gcloud -q compute tpus delete --zone="$ZONE" "$TPU_NAME"
gcloud -q compute instances delete --delete-disks='all' --zone="$ZONE" "$INSTANCE"
gcloud -q compute addresses delete --global "$ADDRESSES"
gcloud -q compute firewall-rules delete "$FIREWALL"
gcloud -q compute networks delete "$NETWORK"
