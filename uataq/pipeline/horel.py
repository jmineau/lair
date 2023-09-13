#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:32:25 2023

@author: James Mineau (James.Mineau@utah.edu)

Module of functions for handing horel-group operations
"""

import os
import pandas as pd
import tables as pytbls
import numpy as np

from lair.config import HOREL_DIR, HOREL_TRAX_DIR, TRAX_PILOT_DIR, vprint
from lair.uataq.pipeline.preprocess import preprocessor
from lair.utils.records import DataFile, filter_files

COL_MAPPER = {
    'EPOCHTIME': 'Time_UTC',

    # CR1000
    'GELV': 'Altitude_m',
    'GLAT': 'Lati_deg',
    'GLON': 'Long_deg',
    'GTIM': 'GPS_Time_UTC',
    'NSAT': 'NSat',
    'TICC': 'Panel_T_C',
    'TRNR': 'Train_RH',
    'TRNT': 'Train_T_C',
    'VOLT': 'Battery_Voltage_V',

    # 2B
    'FL2B': 'Flow_ccmin',
    'OZNE': 'O3_ppb',
    'PS2B': 'Cavity_P_hPa',
    'TC2B': 'Cavity_T_C',

    # ESAMPLER
    'ERRR': 'Status',
    'FLOW': 'Flow_Lmin',
    'INRH': 'Cavity_RH_pct',
    'ITMP': 'Ambient_T_C',
    'PM25': 'PM2.5_ugm3',
    'PRES': 'Ambient_P_hPa'
}


@preprocessor
def get_files(site, instrument, time_range=None):
    files = []

    for parent_dir in [TRAX_PILOT_DIR, HOREL_TRAX_DIR]:
        data_dir = os.path.join(parent_dir, instrument.lower())
        for file in os.listdir(data_dir):
            if file.startswith(site.upper()):
                file_path = os.path.join(data_dir, file)

                date_str = file[6: 13].replace('_', '-')
                period = pd.Period(date_str, freq='M')

                files.append(DataFile(file_path, period))

    return filter_files(files, time_range)


def _parse(file):

    vprint(f'Parsing {os.path.relpath(file, HOREL_DIR)}')

    df = pd.read_hdf(file, key='obsdata/observations')

    # Rename horel-group columns according to COL_MAPPER
    df.rename(columns=COL_MAPPER, inplace=True)

    # Format time
    df['Time_UTC'] = pd.to_datetime(df.Time_UTC, unit='s')

    # Convert horel-group NoData to np.nan
    df.replace(-9999, np.nan, inplace=True)

    return df


def _parse_h5(file, key='obsdata/observations'):
    with pytbls.open_file(file, mode='r') as f:
        table = f.root[key]
        data = table.read()
