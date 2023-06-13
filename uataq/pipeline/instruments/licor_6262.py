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
from ..preprocess import preprocessor
from utils.records import filter_files


INSTRUMENT = 'licor_6262'
data_config = data_config[INSTRUMENT]


@preprocessor
def get_files(site, lvl, time_range=None):
    data_dir = os.path.join(DATA_DIR, site, INSTRUMENT, lvl)

    files = [(os.path.join(data_dir, file), file[:7])
             for file in os.listdir(data_dir) if file.endswith('dat')]

    return sorted(filter_files(files, '%Y_%m', time_range))


@preprocessor
def read_obs(site, specie='CO2', lvl='calibrated', time_range=None):
    assert specie == 'CO2'

    names = data_config[lvl]['col_names']
    types_R = data_config[lvl]['col_types']
    types = {name: r2py_types[t] for name, t in zip(names, types_R)}

    time_col = next((col for col in names if 'TIME' in col.upper()), None)

    files = get_files(site, lvl, time_range)

    dfs = []
    for file in files:
        df = pd.read_csv(file, names=names, dtype=types, header=0,
                         on_bad_lines='skip', na_values=['NAN'])

        dfs.append(df)

    df = pd.concat(dfs)

    # Set time as index and filter to time_range
    df = df.set_index(time_col).sort_index()
    df = df.loc[time_range[0]: time_range[1]]

    return df
