"""
Config module for lair package
"""

import os
import pandas as pd


###############
# Directories #
###############

#: CHPC Home
HOME = '/uufs/chpc.utah.edu/common/home'

# -- LAIR GROUP -- #
#: Lin Group Directory
GROUP_DIR = os.path.join(HOME, 'lin-group11', 'group_data')

#: Lin Group Inventory Directory
INVENTORY_DIR = os.path.join(GROUP_DIR, 'inventories')

#: Lin Group Met-Field Directory
MET_DIR = os.path.join(GROUP_DIR, 'NOAA-ARL_formatted_metfields')

#: Lin Group Spatial File Directory
SPATIAL_DIR = os.path.join(GROUP_DIR, 'spatial')

# -- HOREL GROUP -- #
#: Horel Group Directory
HOREL_DIR = os.path.join(HOME, 'horel-group')

#: MesoWest Directory
MESOWEST_DIR = os.path.join(HOREL_DIR, 'oper/mesowest')

#: User STILT Directory
STILT_DIR = os.getenv('STILT_DIR')

# LAIR
LAIR_DIR = os.path.dirname(__file__)

#: User Cache Directory
CACHE_DIR = os.getenv('LAIR_CACHE_DIR',
                      os.path.join(os.path.expanduser('~'), '.cache', 'lair'))
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

##########
# PANDAS #
##########

#: Pandas copy-on-write
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
