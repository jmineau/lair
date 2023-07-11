#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:42:44 2023

@author: James Mineau (James.Mineau@utah.edu)

Module of uataq pipeline functions for teledyne t500u instrument
"""

import os
import pandas as pd

from config import DATA_DIR, vprint
from ..preprocess import preprocessor
from utils.records import DataFile, filter_files, parallelize_file_parser

INSTRUMENT = 'teledyne_t500u'

NAMES = ['time', 'phase_t_c', 'bench_phase_s', 'meas_l_mm', 'aref_l_mm',
         'samp_pres_inhga', 'samp_temp_c', 'bench_t_c', 'box_t_c', 'no2_slope',
         'no2_offset_mv', 'no2_ppb', 'no2_std_ppb', 'mf_t_c', 'test_mv']


@preprocessor
def get_files(site, lvl='raw', time_range=None):
    # There are only raw files for t500u's
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

    df = pd.read_csv(file, names=NAMES, header=0,
                     on_bad_lines='skip', na_values=['XXXX'])

    # Format time
    df.rename(columns={'time': 'Time_UTC'}, inplace=True)
    df['Time_UTC'] = pd.to_datetime(df.Time_UTC, errors='coerce')

    return df


@preprocessor
def read_obs(site, specie='NO2', lvl='raw', time_range=None, num_processes=1):
    assert specie == 'NO2'

    vprint(f'Reading {lvl} observations collected by Teledyne T500u at {site}')

    files = get_files(site, lvl, time_range)

    read_files = parallelize_file_parser(_parse, num_processes=num_processes)
    df = pd.concat(read_files(files))

    # Set time as index and filter to time_range
    df = df.dropna(subset='Time_UTC').set_index('Time_UTC').sort_index()
    df = df.loc[time_range[0]: time_range[1]]

    return df
