#!/usr/bin/env bash

declare -r DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -euo pipefail

"$DIR/preflight.bash"

gcloud compute networks create "$NETWORK"
gcloud compute firewall-rules create --allow='tcp:22,tcp:3389,tcp:443,tcp:80,icmp' \
                                     --network "$NETWORK" \
                                     "$FIREWALL"
gcloud compute addresses create --addresses="$RANGE" \
                                --network="$NETWORK" \
                                --purpose='VPC_PEERING' \
                                --prefix-length="$CIDR" \
                                --global \
                                "$ADDRESSES"

if [[ "$OS"=="Ubuntu" ]]; then
#                                   --boot-disk-size='500GB' \
    gcloud compute instances create --image-project='ubuntu-os-cloud' \
                                    --image-family='ubuntu-2004-lts' \
                                    --machine-type='n1-standard-2' \
                                    --network="$NETWORK" \
                                    --scopes='cloud-platform' \
                                    "$INSTANCE"
else
#                                   --boot-disk-size='500GB' \
    gcloud compute instances create --image-project='debian-cloud' \
                                    --image-family='debian-10' \
                                    --machine-type='n1-standard-2' \
                                    --network="$NETWORK" \
                                    --scopes='cloud-platform' \
                                    "$INSTANCE"
fi

gcloud compute tpus create --accelerator-type="$ACCELERATOR" \
                           --description="$TPU_NAME" \
                           --network="$NETWORK" \
                           --preemptible \
                           --range="$RANGE" \
                           --version="$VERSION" \
                           --zone="$ZONE" \
                           "$TPU_NAME"
