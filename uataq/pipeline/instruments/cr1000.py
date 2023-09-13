#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:27:03 2023

@author: James Mineau (James.Mineau@utah.edu)

Module of uataq pipeline functions for Horel-group's cr1000
"""

from functools import partial
import pandas as pd

from ....config import vprint
from .. import horel
from ..preprocess import preprocessor
from ....utils.records import parallelize_file_parser


get_files = partial(horel.get_files, instrument='cr1000')

_parse = horel._parse


@preprocessor
def read_obs(site, specie='GPS', time_range=None, num_processes=1):

    vprint(f'Reading observations collected by CR1000 at {site}')

    files = get_files(site, time_range=time_range)

    read_files = parallelize_file_parser(_parse, num_processes=num_processes)

    df = pd.concat(read_files(files))

    df = df.dropna(subset='Time_UTC').set_index('Time_UTC').sort_index()
    df = df.loc[time_range[0]: time_range[1]]

    return df
