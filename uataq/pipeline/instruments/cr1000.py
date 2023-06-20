#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:27:03 2023

@author: James Mineau (James.Mineau@utah.edu)

Module of uataq pipeline functions for Horel-group's cr1000
"""

from functools import partial
import pandas as pd

from .. import horel
from ..preprocess import preprocessor


get_files = partial(horel.get_files, instrument='cr1000')


@preprocessor
def read_obs(site, specie='GPS', time_range=None):
    files = get_files(site, time_range=time_range)

    df = pd.concat([horel.read_file(file) for file in files]).sort_index()

    return df
