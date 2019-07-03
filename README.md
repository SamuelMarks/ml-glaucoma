ml_glaucoma
===========
A Glaucoma diagnosing CNN

## Install dependencies

    pip install -r requirements.txt

## Install package

    pip install .

## CLI usage

    $ python -m ml_glaucoma --help

    usage: __main__.py [-h] [--version] {download,cnn,parser,v2} ...
    
    CLI for a Glaucoma diagnosing CNN and preparing data for such
    
    positional arguments:
      {download,cnn,parser,v2}
        download            Download required data
        cnn                 Convolutional Neural Network runner
        parser              Show metrics from output. Default: per epoch
                            sensitivity & specificity.
        v2                  Alternative CLI parser
    
    optional arguments:
      -h, --help            show this help message and exit
      --version             show program's version number and exit


### `download`

    $ python -m ml_glaucoma download --help

    usage: __main__.py download [-h] -d DOWNLOAD_DIR [-f]

    optional arguments:
      -h, --help            show this help message and exit
      -d DOWNLOAD_DIR, --download-dir DOWNLOAD_DIR
                            Directory to store precompiled CNN nets
      -f, --force           Force recreation of precompiled CNN nets

### `cnn`

    $ python -m ml_glaucoma cnn --help
    usage: __main__.py cnn [-h] [-b BATCH_SIZE] [-n NUM_CLASSES] [-e EPOCHS] -m
                       MODEL_NAME -s PREPROCESS_TO -d DOWNLOAD_DIR
                       [-t TRANSFER_MODEL] [--dropout DROPOUT] [-p PIXELS]
                       [--tensorboard-log-dir TENSORBOARD_LOG_DIR]
                       [--optimizer OPTIMIZER] [--loss LOSS]
                       [--architecture ARCHITECTURE] [--metrics METRICS]
                       [--split-dir SPLIT_DIR]
                       [--bmes123-pardir BMES123_PARDIR]
                       [--class-mode {categorical,binary,sparse}] [--lr LR]
                       [--max-imgs MAX_IMGS]

    optional arguments:
      -h, --help            show this help message and exit
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            Batch size
      -n NUM_CLASSES, --num-classes NUM_CLASSES
                            Number of classes
      -e EPOCHS, --epochs EPOCHS
                            Number of epochs
      -m MODEL_NAME, --model MODEL_NAME
                            Filename for h5 trained model file
      -s PREPROCESS_TO, --save PREPROCESS_TO
                            Save h5 file of dataset, following preprocessing
      -d DOWNLOAD_DIR, --download-dir DOWNLOAD_DIR
                            Directory to store precompiled CNN nets
      -t TRANSFER_MODEL, --transfer-model TRANSFER_MODEL
                            Transfer model. Currently any one of:
                            `keras.application`, e.g.: "vgg16"; "resnet50"
      --dropout DROPOUT     Dropout (0,1,2,3 or 4)
      -p PIXELS, --pixels PIXELS
                            Pixels. E.g.: 400 for 400px * 400px
      --tensorboard-log-dir TENSORBOARD_LOG_DIR
                            Enables Tensorboard integration and sets its log dir
      --optimizer OPTIMIZER
      --loss LOSS
      --architecture ARCHITECTURE
                            Current options: unet; for U-Net architecture
      --metrics METRICS     precision_recall or btp
      --split-dir SPLIT_DIR
                            Place to create symbolic links for train, test,
                            validation split
      --bmes123-pardir BMES123_PARDIR
                            Parent folder of BMES123 folder
      --class-mode {categorical,binary,sparse}
                            Determines the type of label arrays that are returned
      --lr LR, --learning-rate LR
                            Learning rate of optimiser
      --max-imgs MAX_IMGS   max_imgs

### `parser`
You can pipe or include a filename.

    $ python -m ml_glaucoma parser --help

    usage: __main__.py parser [-h] [--threshold THRESHOLD] [--top TOP] [--by-diff]
                              [infile]

    positional arguments:
      infile                File to work from. Defaults to stdin. So can pipe.

    optional arguments:
      -h, --help            show this help message and exit
      --threshold THRESHOLD
                            E.g.: 0.7 for sensitivity & specificity >= 70%
      --top TOP             Show top k results
      --by-diff             Sort by lowest difference between sensitivity &
                            specificity

# V2 Project Structure

Training/validation scripts are provided in `bin` and each call a function defined in `ml_glaucoma.runners`. We aim to provide highly-configurable runs, but the main parts to consider are:

* `problem`: the dataset, loss and metrics used during training
* `model_fn`: the function that takes one or more `tf.keras.layers.Input`s and returns a learnable keras model.

`model_fn`s are configured using using a forked [TF2.0 compatible gin-config](https://github.com/jackd/gin-config/tree/tf2) (awaiting on [this PR](https://github.com/google/gin-config/pull/17) before reverting to the [google version](https://github.com/google/gin-config.git). See example configs in `model_configs` and the [gin user guide](https://github.com/google/gin-config/blob/master/docs/index.md).

## Command usage:
```bash
$ python -m ml_glaucoma v2 --help

usage: __main__.py v2 [-h] {download,vis,train,evaluate} ...

positional arguments:
  {download,vis,train,evaluate}
    download            Download and prepare required data
    vis                 Download and prepare required data
    train               Train model
    evaluate            Evaluate model

optional arguments:
  -h, --help            show this help message and exit
```

## Example usage:

```bash
python -m ml_glaucoma v2 vis --dataset=refuge
python -m ml_glaucoma v2 train \
  --model_file 'model_configs/dc.gin'  \
  --model_param 'import ml_glaucoma.gin_keras' 'dc0.kernel_regularizer=@tf.keras.regularizers.l2()' 'tf.keras.regularizers.l2.l = 1e-2' \
  --model_dir /tmp/ml_glaucoma/dc0-reg \
  -m BinaryAccuracy AUC \
  -pt 0.1 0.2 0.5 -rt 0.1 0.2 0.5 \
  --use_inverse_freq_weights
# ...
tensorboard --logdir=/tmp/ml_glaucoma
```

## Tensorflow Datasets

The main `Problem` implementation is backed by [tensorflow_datasets](https://github.com/tensorflow/datasets). This should manage dataset downloads, extraction, sha256 checks, on-disk shuffling/sharding and other best practices. Consequently it takes slightly longer to process initially, but the benefits in the long run are worth it.

## BMES Initialization

The current implementation leverages the existing `ml_glaucoma.utils.get_data` method to separate files. This uses `tf.contrib` so requires `tf < 2.0`. It can be run using the `--bmes_init` flag in `bin/__main__.py`. This must be run prior to the standard `tfds.DatasetBuilder.download_and_prepare` which is run automatically if necessary. Once the `tfds` files have been generated, the original `get_data` directories are no longer required.

If the test/train/validation split here is just a random split, this could be done more easily by creating a single `tfds` split and using `tfds.Split.subsplit` - see [this post](https://www.tensorflow.org/datasets/splits).

## Status

* Automatic model saving/loading via modified `ModelCheckpoint`.
* Automatic tensorboard updates (fairly hacky interoperability with `ModelCheckpoint` to ensure restarted training runs have the appropriate step count).
* Loss re-weighting according to inverse class frequency (`TfdsProblem.use_inverse_freq_weights`).
* Only `dc0` model verified to work. `dc1`, `unet` and `tf.keras.applications` implemented but untested.
* Only `refuge` dataset implemented, and only tested the classification task.
* BMES dataset: currently requires 2-stage preparation: `bmes_init` which is based on `ml_glaucoma.utils.get_data` and the standard `tfds.DatasetBuilder.download_and_prepare`. The first stage will only be run if `--bmes_init` is used in `__main__.py`.
