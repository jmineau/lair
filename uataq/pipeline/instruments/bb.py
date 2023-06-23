#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:38:05 2023

@author: James Mineau (James.Mineau@utah.edu)

Module of uataq pipeline functions for 2B instrument

LAIR only uses 2b's for mobile platforms!
"""

import os
import pandas as pd
from pandas.errors import ParserError

from config import DATA_DIR
from .. import horel
from ..preprocess import preprocessor
from utils.records import DataFile, filter_files, parallelize_file_parser


INSTRUMENT = '2b'


@preprocessor
def get_files(site, lvl='raw', time_range=None, use_lin_group=False):

    if use_lin_group:

        files = []

        data_dir = os.path.join(DATA_DIR, site, '2bo3', lvl)

        if lvl == 'raw':
            date_format = '%Y_%m_%d'

            if site == 'trx01':
                date_slicer = slice(3, 13)
            elif site == 'trx02':
                date_slicer = slice(10)
        else:
            date_format = '%Y_%m'
            date_slicer = slice(7)

        for file in os.listdir(data_dir):
            if file.endswith('dat'):
                file_path = os.path.join(data_dir, file)
                date = pd.to_datetime(file[date_slicer], format=date_format)

                files.append(DataFile(file_path, date))

        return filter_files(files, time_range)

    return horel.get_files(site, INSTRUMENT, time_range)


def _parse(file, lvl, use_lin_group):

    names = ['Time_UTC', 'O3_ppb', 'Cavity_T_C', 'Cavity_P_hPa', 'Flow_ccmin']
    if lvl == 'raw':
        names += ['Date_MTN', 'Time_MTN']
    else:
        names.append('QAQC_Flag')

    if use_lin_group:
        # Read files from lin-group9
        try:
            df = pd.read_csv(file, on_bad_lines='skip', names=names,
                             dtype=str)

        except (ParserError, UnicodeDecodeError):
            return None

        # Format time
        df['Time_UTC'] = pd.to_datetime(df.Time_UTC, errors='coerce',
                                        format='ISO8601')

        if lvl == 'raw':
            df['Time_MTN_2B'] = pd.to_datetime(df.Date_MTN + df.Time_MTN,
                                               errors='coerce',
                                               format='%d/%m/%y%H:%M:%S')
            df.drop(columns=['Date_MTN', 'Time_MTN'], inplace=True)
            names = names[:-2]

        for col in names[2:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    else:
        # Read files from horel-group
        df = horel._parse(file)

        if lvl != 'raw':
            # Create QAQC_Flag column
            df['QAQC_Flag'] = 0


@preprocessor
def read_obs(site, specie='O3', lvl='raw', time_range=None,
             use_lin_group=False, num_processes=1):
    assert specie == 'O3'

    files = get_files(site, lvl=lvl, time_range=time_range,
                      use_lin_group=use_lin_group)

    read_files = parallelize_file_parser(_parse, num_processes=num_processes)
    df = pd.concat(read_files(files, lvl=lvl, use_lin_group=use_lin_group))

    # Set time as index and filter to time_range
    df = df.dropna(subset='Time_UTC').set_index('Time_UTC').sort_index()
    df = df.loc[time_range[0]: time_range[1]]

    if lvl != 'raw':
        # Drop rows without an O3 reading
        df.dropna(subset='O3_ppb', inplace=True)

        # Set flag to -1 (manually removed) if less than 0 (impossible)
        df.QAQC_Flag[df.O3_ppb < 0] = -1

    return df
