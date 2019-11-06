Proposal
========

## Step 0
Construct a dataset & a CLI for training on dataset

## Step 1
Using enumeration of options—like callbacks, loss, transfer, and optimizers—derive all possible permutations into a `pipeline`.

Let's start with just a `transfer_pipeline`, over all the possible transfer models (ResNet, EfficientNet, MobileNet, &etc).

## Step 2
Run `transfer_pipeline` on 3 different GPU-powered servers.

## Step 3
Process results of all `transfer_pipeline` runs.

---

## Step 4
Decide on next `pipeline` of experiments, e.g.:
- `loss_pipeline`
- `optimizer_pipeline`
- `callback_pipeline`
