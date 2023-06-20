#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:40:57 2023

@author: James Mineau (James.Mineau@utah.edu)

Module of uataq pipeline functions for metone esampler instrument
"""

import os
import pandas as pd

from config import DATA_DIR, data_config
from .. import horel
from ..preprocess import preprocessor
from utils.records import DataFile, filter_files

INSTRUMENT = 'metone_es642'
data_config = data_config[INSTRUMENT]


@preprocessor
def get_files(site, lvl, time_range=None, use_lin_group=False):
    if use_lin_group:
        files = []
        if site == 'trx02':
            instrument = 'metone'  # there is also a met dir
            date_slicer = slice(10)
            date_format = '%Y_%m_%d'
        else:
            instrument = INSTRUMENT
            date_slicer = slice(7)
            date_format = '%Y_%m'

        data_dir = os.path.join(DATA_DIR, site, instrument, lvl)

        for file in os.listdir(data_dir):
            if file.endswith('dat'):
                file_path = os.path.join(data_dir, file)
                date = pd.to_datetime(file[date_slicer], format=date_format)

                files.append(DataFile(file_path, date))

        return filter_files(files, time_range)

    return horel.get_files(site, 'esampler', time_range)


@preprocessor
def read_obs(site, specie='PM_25', lvl='raw', time_range=None,
             use_lin_group=False):

    if site in ['dbk', 'sug', 'wbb']:
        use_lin_group = True

    files = get_files(site, lvl, time_range, use_lin_group)

    dfs = []
    for file in files:
        if use_lin_group:
            # Read from lin-group
            if site == 'trx02' or (site == 'wbb' and lvl == 'raw'):

                # Read pi data
                names = ['Timestamp', 'PM_25_Avg', 'Flow_Avg', 'Temp_Avg',
                         'RH_Avg', 'BP_Avg', 'Status', 'Checksum']

                df = pd.read_csv(file, names=names, header=None,
                                 on_bad_lines='skip')

            else:
                names = data_config[lvl]['col_names']

                df = pd.read_csv(file)

            # Set time as index and sort
            time_col = next((col for col in names
                             if 'TIME' in col.upper()), None)

            df[time_col] = pd.to_datetime(df[time_col], errors='coerce',
                                          format='ISO8601')
            df.dropna(subset=time_col, inplace=True)
            df = df.set_index(time_col)

        else:
            # Read from horel-group
            df = horel.read_file(file)

            if lvl != 'raw':
                # Create QAQC_Flag column
                df['QAQC_Flag'] = 0

        dfs.append(df)

    df = pd.concat(dfs).sort_index()

    # Filter to time_range
    df = df.loc[time_range[0]: time_range[1]]

    return df
