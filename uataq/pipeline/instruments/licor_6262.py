#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 12:13:14 2023

@author: James Mineau (James.Mineau@utah.edu)

Module of uataq pipeline functions for licor 6262 instrument
"""

import os
import pandas as pd

from config import DATA_DIR, data_config, r2py_types
from utils.records import filter_files


INSTRUMENT = 'licor_6262'
data_config = data_config[INSTRUMENT]


def get_files(site, lvl, time_range=None):
    data_dir = os.path.join(DATA_DIR, site, INSTRUMENT, lvl)

    files = [(os.path.join(data_dir, file), file[:7])
             for file in os.listdir(data_dir) if file.endswith('dat')]

    return sorted(filter_files(files, '%Y_%m', time_range))


def read_raw(site, lvl, time_range=None):


    names = data_config['raw']['col_names']
    types_R = data_config['raw']['col_types']
    types = {name: r2py_types[t] for name, t in zip(names, types_R)}

    numeric_cols = [name for name, t in zip(names, types_R)
                    if t == 'd']

    files = get_files(site, lvl, time_range)

    dfs = []
    for file in files:
        print(file)
        df = pd.read_csv(file, names=names, dtype=types, header=0,
                         on_bad_lines='skip', na_values=['NAN'])

        dfs.append(df)

    df = pd.concat(dfs)

    return df


def read_qaqc(site):
    pass


def read_calibrated(site):
    pass


def read_obs(site, species=['co2'], lvl='calibrated', time_range=None,
             **kwargs):
    if lvl == 'raw':
        return read_raw(site)
    elif lvl == 'qaqc':
        return read_qaqc(site)
    elif lvl == 'calibrated':
        return read_calibrated(site)
