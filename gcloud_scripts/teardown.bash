#!/usr/bin/env bash

if [ -z "$NETWORK" ]; then
  printf '$NETWORK not defined; try including all environment variables specified in README.md\n' >&2
  exit 2
fi

gcloud compute tpus delete --zone="$ZONE" "$TPU_NAME"
gcloud compute instances delete --delete-disks='all' --zone="$ZONE" "$INSTANCE"
gcloud compute addresses delete --global "$ADDRESSES"
gcloud compute firewall-rules delete "$FIREWALL"
gcloud compute networks delete "$NETWORK"
