ml_glaucoma
===========
A Glaucoma diagnosing CNN

## Install dependencies

    pip install -r requirements.txt

## Install package

    pip install .

## CLI usage

    $ python -m ml_glaucoma --help

    usage: __main__.py [-h] [--version] {data,download,cnn} ...

    CLI for a Glaucoma diagnosing CNN and preparing data for such
    
    positional arguments:
      {data,download,cnn}
        data               Data preprocessing runner
        download           Download required data
        cnn                Convolutional Neural Network runner
    
    optional arguments:
      -h, --help           show this help message and exit
      --version            show program's version number and exit

### `data`

    $ python -m ml_glaucoma data --help
    
    usage: __main__.py data [-h] -s SAVE [-f]
    
    optional arguments:
      -h, --help            show this help message and exit
      -s SAVE, --save SAVE  Save h5 file of dataset
      -f, --force           Force new h5 file of dataset being created

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
                           MODEL_NAME -s SAVE_TO -d DOWNLOAD_DIR
    
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
      -s SAVE_TO, --save SAVE_TO
                            Save h5 file of dataset
      -d DOWNLOAD_DIR, --download-dir DOWNLOAD_DIR
                            Directory to store precompiled CNN nets
