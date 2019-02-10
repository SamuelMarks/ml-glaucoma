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
