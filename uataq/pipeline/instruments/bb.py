#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:38:05 2023

@author: James Mineau (James.Mineau@utah.edu)

Module of uataq pipeline functions for 2B instrument

LAIR only uses 2b's for mobile platforms!
"""

import numpy as np
import os
import pandas as pd
from pandas.errors import ParserError

from config import DATA_DIR, HOREL_TRAX_DIR, TRAX_PILOT_DIR
from ..preprocess import preprocessor
from utils.records import filter_files


INSTRUMENT = '2b'

COL_MAPPER = {
    'FL2B': 'Flow_ccmin',
    'OZNE': 'O3_ppb',
    'PS2B': 'Cavity_P_hPa',
    'TC2B': 'Cavity_T_C'}


@preprocessor
def get_files(site, lvl='raw', time_range=None, use_lin_group=False):
    if use_lin_group:

        data_dir = os.path.join(DATA_DIR, site, '2bo3', lvl)

        if lvl == 'raw':
            datetime_format = '%Y_%m_%d'

            if site == 'trx01':
                date_slicer = slice(3, 13)
            elif site == 'trx02':
                date_slicer = slice(10)
        else:
            datetime_format = '%Y_%m'
            date_slicer = slice(7)

        files_dates = [(os.path.join(data_dir, file), file[date_slicer])
                       for file in os.listdir(data_dir)
                       if file.endswith('dat')]

    else:
        datetime_format = '%Y_%m'
        date_slicer = slice(6, 13)

        files_dates = []
        for parent_dir in [TRAX_PILOT_DIR, HOREL_TRAX_DIR]:
            dir_path = os.path.join(parent_dir, '2b')
            for file in os.listdir(dir_path):
                if file.startswith(site.upper()):
                    file_path = os.path.join(dir_path, file)
                    date = file[date_slicer]
                    files_dates.append((file_path, date))

    return sorted(filter_files(files_dates, datetime_format, time_range))


@preprocessor
def read_obs(site, specie='O3', lvl='raw', time_range=None,
             use_lin_group=False):
    assert specie == 'O3'

    files = get_files(site, lvl=lvl, time_range=time_range,
                      use_lin_group=use_lin_group)

    names = ['Time_UTC', 'O3_ppb', 'Cavity_T_C', 'Cavity_P_hPa', 'Flow_ccmin']

    dfs = []
    for file in files:
        if use_lin_group:
            # Read files from lin-group9
            try:
                df = pd.read_csv(file, on_bad_lines='skip',
                                 names=names+['Date_MTN', 'Time_MTN'])
            except (ParserError, UnicodeDecodeError):
                continue

            # Format time
            df['Time_UTC'] = pd.to_datetime(df.Time_UTC, errors='coerce',
                                            format='ISO8601')

            for col in names[1:]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows where any column is missing data.
            #   Assume that if Pi messed up one column, then none of the
            #   columns can be trusted.
            df.dropna(how='any', inplace=True)

        else:
            # Read files from horel-group
            df = pd.read_hdf(file, key='obsdata/observations')
            df['Time_UTC'] = pd.to_datetime(df.EPOCHTIME, unit='s')
            df = df.rename(columns=COL_MAPPER)
            df.replace(-9999, np.nan, inplace=True)

        dfs.append(df)

    df = pd.concat(dfs)

    # Set time as index and filter to time_range
    df = df.set_index('Time_UTC').sort_index()
    df = df.loc[time_range[0]: time_range[1]]

    # Drop horel cols that are not in names
    names_in_data = [name for name in names if name in df.columns]

    return df.loc[:, names_in_data]
