from os import path

SAVE_FORMAT = 'h5'
# SAVE_FORMAT.__doc__ =
"""
Either 'tf' or 'h5', indicating whether to save the model
        to Tensorflow SavedModel or HDF5. Meant to default to 'tf' in TF 2.X, and 'h5' in TF 1.X.
      ImportError: If save format is hdf5, and h5py is not available
"""

SAVE_FORMAT_WITH_SEP = '{}{}'.format(path.extsep, SAVE_FORMAT)

IMAGE_RESOLUTION = 224, 224

del path
