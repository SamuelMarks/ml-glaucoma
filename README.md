ml_glaucoma
===========
A Glaucoma diagnosing CNN

## Install dependencies

    pip install -r requirements.txt

## Install package

    pip install .

## CLI usage

    $ python -m ml_glaucoma --help

    usage: __main__.py [-h] [-b BATCH_SIZE] [-n NUM_CLASSES] [-e EPOCHS] -m MODEL
                       -o OUTPUT [--version]
    
    CLI for a Glaucoma diagnosing CNN
    
    optional arguments:
      -h, --help            show this help message and exit
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            Batch size
      -n NUM_CLASSES, --num-classes NUM_CLASSES
                            Number of classes
      -e EPOCHS, --epochs EPOCHS
                            Number of epochs
      -m MODEL, --model MODEL
                            Filename for h5 trained model file
      -o OUTPUT, --output OUTPUT
                            Output in hdf5 format to this filename
      --version             show program's version number and exit
