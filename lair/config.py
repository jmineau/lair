"""
lair.config
~~~~~~~~~~~~

Config module for lair package
"""

import os
import pandas as pd
import re


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

# Cache
class _CacheDir:
    global CACHE_DIR
    CACHE_DIR = "/uufs/chpc.utah.edu/common/home/lin-group23/jkm/lair_cache"

    @property
    def directory(self):
        return CACHE_DIR
    
    @directory.getter
    def directory(self):
        if CACHE_DIR is None:
            cache_dir = input("Enter the path to your lair cache directory: ")
            # Read the content of the file
            with open(__file__, 'r') as file:
                content = file.read()

            # Use regex to find and replace CACHE_DIR = ""
            content = re.sub(r'CACHE_DIR\s*=\s*None', f'CACHE_DIR = "{cache_dir}"', content)

            # Write the modified content back to the file
            with open(__file__, 'w') as file:
                file.write(content)

            return cache_dir
        else:
            return CACHE_DIR
CACHE_DIR = _CacheDir().directory


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
