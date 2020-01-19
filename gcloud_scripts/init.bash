#!/usr/bin/env bash

if [ -z "$NETWORK" ]; then
  printf '$NETWORK not defined; try including all environment variables specified in README.md\n' >&2
  exit 2
fi

gcloud compute networks create "$NETWORK"
gcloud compute firewall-rules create --network "$NETWORK" --allow tcp:22,tcp:3389,tcp:443,tcp:80,icmp \
                                     "$FIREWALL"
gcloud compute addresses create --global --purpose='VPC_PEERING' \
                                --addresses="$RANGE" --prefix-length="$CIDR" \
                                --network="$NETWORK" \
                                "$ADDRESSES"
gcloud compute instances create --machine-type='n1-standard-4' --boot-disk-size='500GB' \
                                --image-project='debian-cloud' --image-family='debian-10' \
                                --scopes=cloud-platform --network="$NETWORK" \
                                "$INSTANCE"
gcloud compute tpus create --zone="$ZONE" --description="$TPU_NAME" \
                           --accelerator-type="$ACCELERATOR" --version="$VERSION" \
                           --network="$NETWORK" --range="$RANGE" \
                           "$TPU_NAME"
