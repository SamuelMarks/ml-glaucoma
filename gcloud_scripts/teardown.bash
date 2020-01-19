#!/usr/bin/env bash

set -euo pipefail

if [ -z "$NETWORK" ]; then
  printf '$NETWORK not defined; try including all environment variables specified in README.md\n' >&2
  exit 2
fi

gcloud compute tpus delete --no-user-output-enabled --zone="$ZONE" "$TPU_NAME"
gcloud compute instances delete --no-user-output-enabled --delete-disks='all' --zone="$ZONE" "$INSTANCE"
gcloud compute addresses delete --no-user-output-enabled --global "$ADDRESSES"
gcloud compute firewall-rules delete --no-user-output-enabled "$FIREWALL"
gcloud compute networks delete --no-user-output-enabled "$NETWORK"
