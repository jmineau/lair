#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 12:35:34 2023

@author: James Mineau (James.Mineau@utah.edu)

Module of uataq pipeline functions for magee ae33 instrument
"""

import os
import pandas as pd

from config import DATA_DIR, vprint
from ..preprocess import preprocessor
from utils.records import DataFile, filter_files, parallelize_file_parser

INSTRUMENT = 'magee_ae33'


@preprocessor
def get_files(site='wbb', lvl='raw', time_range=None):
    # There are only raw files for magee
    data_dir = os.path.join(DATA_DIR, site, INSTRUMENT, lvl)

    files = []

    for file in os.listdir(data_dir):
        if file.endswith('csv'):
            file_path = os.path.join(data_dir, file)

            date_str = file[:10].replace('_', '-')
            period = pd.Period(date_str, freq='D')

            files.append(DataFile(file_path, period))

    return filter_files(files, time_range)


def _parse(file):

    vprint(f'Parsing {os.path.relpath(file, DATA_DIR)}')

    df = pd.read_csv(file, skipinitialspace=True, on_bad_lines='skip')

    # Format time
    df.rename(columns={'time': 'Time_UTC'}, inplace=True)
    df['Time_UTC'] = pd.to_datetime(df.Time_UTC, errors='coerce')

    return df


@preprocessor
def read_obs(site='wbb', specie='BC', lvl='raw', time_range=None,
             num_processes=1):
    assert specie == 'BC'

    vprint(f'Reading {lvl} observations collected by magee ae33 at {site}')

    files = get_files(site, lvl, time_range)

    read_files = parallelize_file_parser(_parse, num_processes=num_processes)
    df = pd.concat(read_files(files))

    # Set time as index and filter to time_range
    df = df.dropna(subset='Time_UTC').set_index('Time_UTC').sort_index()
    df = df.loc[time_range[0]: time_range[1]]

    return df
