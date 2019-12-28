#!/usr/bin/env python

from os import path

import pandas as pd

from ml_glaucoma.utils.dr_spoc_data_prep.utils import main


def get_data(dr_spoc_dir, manual_dir):  # type: (str, str) -> pd.DataFrame
    dr_spoc_dir, df, filename2cat, combined_df = main(dr_spoc_dir, manual_dir)

    return combined_df


if __name__ == '__main__':
    get_data(dr_spoc_dir=path.join(path.expanduser('~'),
                                   'OneDrive - The University of Sydney (Students)',
                                   'Fundus Photographs for AI',
                                   'DR SPOC Dataset'))
