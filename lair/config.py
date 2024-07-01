#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 16:47:29 2023

@author: James Mineau (James.Mineau@utah.edu)

Config file for lair package
"""

import os
import pandas as pd


###############
# Directories #
###############

HOME = '/uufs/chpc.utah.edu/common/home'

# LAIR GROUP
GROUP_DIR = os.path.join(HOME, 'lin-group11', 'group_data')

INVENTORY_DIR = os.path.join(GROUP_DIR, 'inventories')
MET_DIR = os.path.join(GROUP_DIR, 'NOAA-ARL_formatted_metfields')
MOBILE_DIR = os.path.join(HOME, 'lin-group7', 'mobile')
SPATIAL_DIR = os.path.join(GROUP_DIR, 'spatial')

# HOREL GROUP
HOREL_DIR = os.path.join(HOME, 'horel-group')
MESOWEST_DIR = os.path.join(HOREL_DIR, 'oper/mesowest')

# STILT
STILT_DIR = os.getenv('STILT_DIR')

# LAIR
LAIR_DIR = os.path.dirname(__file__)


##########
# PANDAS #
##########

# copy-on-write
pandas_CoW = True
pd.options.mode.copy_on_write = pandas_CoW


########
# DATA #
########

# TODO data for valley.py needs to be kept somewhere


###################
# Verbose Printer #
###################

class _Printer:
    global verbose
    verbose = True  # FIXME

    @staticmethod
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs, flush=True)


vprint = _Printer().vprint
