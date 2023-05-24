#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 16:47:29 2023

@author: James Mineau (James.Mineau@utah.edu)

Config file for lair package
"""

from os import path
import pandas as pd


###############
# Directories #
###############

HOME = '/uufs/chpc.utah.edu/common/home'

# UATAQ
MEASUREMENTS_DIR = path.join(HOME, 'lin-group9', 'measurements')
DATA_DIR = path.join(MEASUREMENTS_DIR, 'data')
PIPELINE_DIR = path.join(MEASUREMENTS_DIR, 'pipeline')
CONFIG_DIR = path.join(PIPELINE_DIR, 'config')

# MOBILE
MOBILE_DIR = path.join(HOME, 'lin-group7', 'mobile')
HOREL_TRAX_DIR = path.join(HOME, 'horel-group', 'uutrax')

# LAIR GROUP
GROUP_DIR = path.join(HOME, 'lin-group11', 'group_data')

INVENTORY_DIR = path.join(GROUP_DIR, 'inventories')
MET_DIR = path.join(GROUP_DIR, 'NOAA-ARL_formatted_metfields')


########
# DATA #
########

# UATAQ pipeline config
site_config = pd.read_csv(path.join(CONFIG_DIR, 'site_config.csv'),
                          sep=', ', engine='python', index_col='stid')
data_config = pd.read_json(path.join(CONFIG_DIR, 'data_config.json'))

# TODO data for valley.py needs to be kept somewhere
