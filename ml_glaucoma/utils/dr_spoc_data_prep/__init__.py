#!/usr/bin/env python

from os import path

import pandas as pd

from ml_glaucoma.utils.dr_spoc_data_prep.utils import main


def get_data(root_directory, manual_dir):  # type: (str, str or None) -> pd.DataFrame
    directory, df, filename2cat, combined_df = main(root_directory=root_directory,
                                                    manual_dir=manual_dir)

    return combined_df


if __name__ == '__main__':
    get_data(root_directory=path.join(path.expanduser('~'),
                                      'OneDrive - The University of Sydney (Students)'),
             manual_dir=None)

__all__ = ['get_data']
