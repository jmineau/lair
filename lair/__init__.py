import os

import cf_xarray.units  # must be imported before pint_xarray
import pint
from pint.delegates.formatter._compound_unit_helpers import sort_by_dimensionality
import pint_xarray
from pint_xarray import unit_registry as units

from . import config
from .records import ftp_download, unzip


# Custom pint context to convert fluxes from mass <--> substance
mass_flux = pint.Context('mass_flux')
mass_flux.add_transformation('[substance] / [area] / [time]',
                             '[mass] / [area] / [time]',
                        lambda units, substance, mw: substance * mw)
mass_flux.add_transformation('[mass] / [area] / [time]',
                             '[substance] / [area] / [time]',
                        lambda units, mass, mw: mass / mw)
units.add_context(mass_flux)

# Set default pint sorting function to sort by dimensionality
units.formatter.default_sort_func = sort_by_dimensionality


def setup_ccg_filter():
    """
    Setup the CCG filter module from NOAA GML.
    Downloads the necessary files from the FTP server and unzip them if not already present.
    """
    # Define the path for the CCG filter file
    air_dir = os.path.dirname(__file__)
    ccg_filter_file = os.path.join(air_dir, '_ccg_filter.py')

    # Check if the CCG filter file already exists
    if not os.path.exists(ccg_filter_file):

        remote_zf = 'user/thoning/ccgcrv/ccg_filter.zip'
        zf = os.path.join(air_dir, 'ccg_filter.zip')

        # Download the zip file from the FTP server
        ftp_download('ftp.gml.noaa.gov', remote_zf, air_dir)

        # Unzip the downloaded file
        unzip(zf, air_dir)

        # Cleanup: remove the downloaded zip and unnecessary files
        os.remove(zf)
        os.remove(os.path.join(air_dir, 'ccg_dates.py'))
        os.remove(os.path.join(air_dir, 'ccgcrv.py'))

        # Rename the original file with leading underscore (private module)
        os.rename(os.path.join(air_dir, 'ccg_filter.py'), ccg_filter_file)


setup_ccg_filter()


__version__ = '2025.12.7'
