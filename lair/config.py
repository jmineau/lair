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
if int(pd.__version__.split('.')[0]) < 3:
    # Always on (and no longer settable) from pandas 3.0
    pd.options.mode.copy_on_write = pandas_CoW


########
# DATA #
########

# TODO data for valley.py needs to be kept somewhere


###################
# Verbose Printer #
###################

#: Print progress messages (set ``lair.config.verbose = False`` to silence)
verbose = True  # FIXME


class _Printer:
    @staticmethod
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs, flush=True)


vprint = _Printer().vprint
