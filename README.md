ml_glaucoma
===========
A Glaucoma diagnosing CNN

## Install dependencies

    pip install -r requirements.txt

## Install package

    pip install .

## CLI usage

    $ python -m ml_glaucoma --help

    usage: __main__.py [-h] [--version] {download,cnn,parser} ...

    CLI for a Glaucoma diagnosing CNN and preparing data for such

    positional arguments:
      {download,cnn,parser}
        download            Download required data
        cnn                 Convolutional Neural Network runner
        parser              Show metrics from output. Default: per epoch
                            sensitivity & specificity.

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

Runs are all configured using using a forked [TF2.0 compatible gin-config](https://github.com/jackd/gin-config/tree/tf2) (awaiting on [this PR](https://github.com/google/gin-config/pull/17) before reverting to the [google version](https://github.com/google/gin-config.git). While initially a bit confusing, this is a powerful configuration library that effectively sets default parameters in all functions. See example configs and [user guide](https://github.com/google/gin-config/blob/master/docs/index.md).

Example usage:

```bash
cd ml-glaucoma/bin
python vis.py --gin_file='refuge_cls/base.gin'

# it's generally easier to keep track of things with separate config files
python train.py --gin_file='refuge_cls/dc0-base'
tensorboard --logdir=~/ml_glaucoma_models/refuge_cls

# but you can make your own custom tweaks
# Strongly consider changing the model_dir, otherwise you may have issues
python train.py --gin_file='refuge_cls/dc0-reg-weighted' \
  --gin_param='model_dir="/tmp/customized_dc0-reg-weights"' \
  --gin_param='dc0.dropout_rate=0.3' \
  --gin_param='dc0.dense_units=(64, 32)' \
  --gin_param='optimizer=@tf.keras.optimizers.RMSprop()' \
  --gin_param='tf.keras.optimizers.RMSprop.lr=1e-4' \
  --gin_param='epochs=40'
cat /tmp/customized_dc0-reg-weights/operative_config-0.gin  # see saved config
tensorboard --logdir=/tmp/customized_dc0-reg-weights
```

Note that externally configured functions do not have default arguments changed for functions called from regular `.py` functions. For example, if you have a config

```
# my_config.gin
tf.keras.layers.Dense.kernel_regularizer = @tf.keras.regularizers.l2()
```

You will still see
```
# my_script.py
layer = tf.keras.layers.Dense(units)
print(layer.kernel_regularizer)  # None
```

Where it will have an effect is when you configure another configurable with `tf.keras.layers.Dense`
```
# my_config.gin
tf.keras.layers.Dense.kernel_regularizer = @tf.keras.regularizers.l2()
my_network.penultimate_layer_fn = @tf.keras.layers.Dense
```

```python
def my_network(penultimate_layer_fn):
  layer = penultimate_layer_fn()
  print(layer.kernel_regularizer)  # tf.keras.regularizers.L1L2()
```

The biggest downside to `gin-config` is that stack traces tend to get slightly mangled, and errors in `.gin` file syntax can be tricky to find.

## Tensorflow Datasets

The main `Problem` implementation is backed by [tensorflow_datasets](https://github.com/tensorflow/datasets). This should manage dataset downloads, extraction, sha256 checks, on-disk shuffling/sharding and other best practices. Consequently it takes slightly longer to process initially, but the benefits in the long run are worth it.

## Status

* Automatic model saving/loading via `CheckpointManagerCallback`.
* Automatic tensorboard updates (fairly hacky interoperability with `CheckpointManagerCallback` to ensure restarted training runs have the appropriate step count).
* Loss re-weighting according to inverse class frequency (`TfdsProblem.use_inverse_freq_weights`).
* Only `dc0` model verified to work. `dc1`, `unet` and `tf.keras.applications` implemented but untested.
* Only `refuge` dataset implemented, and only tested the classification task.
* Runners tested with `tf 2.0`. `tf.keras.metrics.AUC` is not available in `tf 1.13` so will raise errors (though `tf 1.14` should, possibly with a few fixes).
* Only `AUC` metric currently configured (though plenty more available - particularly in `tf 1.14` onwards).
