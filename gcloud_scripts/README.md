Google Cloud configuration
==========================
First: install the CLI program for your distribution: https://cloud.google.com/sdk/install

Next, let's define some parameters:
```sh
export REGION='us-central1'
export ZONE='us-central1-f'
export PROJECT_NAME='ml-params-proj'
export PROJECT_ID='ml-params-0'
export NETWORK='ml-params-net'
export FIREWALL='ml-params-firewall'
export ADDRESSES='ml-params-addresses'
export ACCELERATOR='v2-8'
export VERSION='2.1'
export TPU_NAME='tpu0'
export RANGE='192.168.0.0'
export CIDR='16'
export INSTANCE='ml-params-vm0'
```

Then we can setup everything:
```sh
gcloud config set compute/region "$REGION"
gcloud config set compute/zone "$ZONE"
gcloud projects create --name "$PROJECT_NAME" --set-as-default "$PROJECT_ID"
```

Now we can actually create things:
```sh
gcloud compute networks create "$NETWORK"
gcloud compute firewall-rules create --network "$NETWORK" --allow='tcp:22,tcp:3389,tcp:443,tcp:80,icmp' \
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
```

Once done, tear it all down:
```sh
gcloud compute tpus delete --zone="$ZONE" "$TPU_NAME"
gcloud compute instances delete --delete-disks='all' --zone="$ZONE" "$INSTANCE"
gcloud compute addresses delete --global "$ADDRESSES"
gcloud compute firewall-rules delete "$FIREWALL"
gcloud compute networks delete "$NETWORK"
```
