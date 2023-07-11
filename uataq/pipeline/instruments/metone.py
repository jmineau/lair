#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:40:57 2023

@author: James Mineau (James.Mineau@utah.edu)

Module of uataq pipeline functions for metone esampler instrument
"""

import os
import pandas as pd

from config import DATA_DIR, data_config, vprint
from .. import horel
from ..preprocess import preprocessor
from utils.records import DataFile, filter_files, parallelize_file_parser

INSTRUMENT = 'metone_es642'
data_config = data_config[INSTRUMENT]


@preprocessor
def get_files(site, lvl, time_range=None, use_lin_group=False):
    if use_lin_group:
        files = []
        if site == 'trx02':
            instrument = 'metone'  # there is also a met dir
            date_slicer = slice(10)
            file_freq = 'D'
        else:
            instrument = INSTRUMENT
            date_slicer = slice(7)
            file_freq = 'M'

        data_dir = os.path.join(DATA_DIR, site, instrument, lvl)

        for file in os.listdir(data_dir):
            if file.endswith('dat'):
                file_path = os.path.join(data_dir, file)

                date_str = file[date_slicer].replace('_', '-')
                period = pd.Period(date_str, freq=file_freq)

                files.append(DataFile(file_path, period))

        return filter_files(files, time_range)

    return horel.get_files(site, 'esampler', time_range)


def _parse(file, site, lvl, use_lin_group):
    if use_lin_group:
        # Read from lin-group

        vprint(f'Parsing {os.path.relpath(file, DATA_DIR)}')

        if site == 'trx02' or (site == 'wbb' and lvl == 'raw'):

            # Read pi data
            names = ['Timestamp', 'PM_25_Avg', 'Flow_Avg', 'Temp_Avg',
                     'RH_Avg', 'BP_Avg', 'Status', 'Checksum']

            df = pd.read_csv(file, names=names, header=None,
                             on_bad_lines='skip')

        else:
            names = data_config[lvl]['col_names']

            df = pd.read_csv(file)

        # Format time col
        df.rename(columns={'TIMESTAMP': 'Time_UTC'}, inplace=True)
        df['Time_UTC'] = pd.to_datetime(df.Time_UTC, errors='coerce',
                                        format='ISO8601')

    else:
        # Read from horel-group
        df = horel._parse(file)

        if lvl != 'raw':
            # Create QAQC_Flag column
            df['QAQC_Flag'] = 0

    return df


@preprocessor
def read_obs(site, specie='PM_25', lvl='raw', time_range=None,
             use_lin_group=False, num_processes=1):

    vprint(f'Reading {lvl} observations collected by Metone ES642 at {site}')

    if site in ['dbk', 'sug', 'wbb']:
        use_lin_group = True

    files = get_files(site, lvl, time_range, use_lin_group)

    read_files = parallelize_file_parser(_parse, num_processes=num_processes)
    df = pd.concat(read_files(files, site=site, lvl=lvl,
                              use_lin_group=use_lin_group))

    # Set time as index and filter to time_range
    df = df.dropna(subset='Time_UTC').set_index('Time_UTC').sort_index()
    df = df.loc[time_range[0]: time_range[1]]

    return df
