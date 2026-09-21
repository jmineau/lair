import os
from importlib.metadata import PackageNotFoundError, version as _version

import cf_xarray.units  # must be imported before pint_xarray
import pint
import pint_xarray
from pint_xarray import unit_registry as units

# sort_by_dimensionality is private pint API and it moves between releases:
# pint >= 0.26 keeps it in `sorting`, pint <= 0.25 in `_compound_unit_helpers`.
try:
    from pint.delegates.formatter.sorting import sort_by_dimensionality  # pyrefly: ignore[missing-import]
except ImportError:
    try:
        from pint.delegates.formatter._compound_unit_helpers import (  # pyrefly: ignore[missing-import]
            sort_by_dimensionality,
        )
    except ImportError:
        sort_by_dimensionality = None

from . import config
from .records import ftp_download, unzip

try:
    __version__ = _version("lair")  # set by setuptools-scm from git tags
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0+unknown"


# Custom pint context to convert fluxes from mass <--> substance
mass_flux = pint.Context("mass_flux")
mass_flux.add_transformation(
    "[substance] / [area] / [time]",
    "[mass] / [area] / [time]",
    # pint types the callback narrowly; (units, value, mw) is what it calls
    # pyrefly: ignore[bad-argument-type]
    lambda units, substance, mw: substance * mw,
)
mass_flux.add_transformation(
    "[mass] / [area] / [time]",
    "[substance] / [area] / [time]",
    # pyrefly: ignore[bad-argument-type]
    lambda units, mass, mw: mass / mw,
)
units.add_context(mass_flux)

# Set default pint sorting function to sort by dimensionality.
# Cosmetic only: if pint moves the helper again, keep pint's own order.
if sort_by_dimensionality is not None:
    units.formatter.default_sort_func = sort_by_dimensionality


def setup_ccg_filter():
    """
    Setup the CCG filter module from NOAA GML.
    Downloads the necessary files from the FTP server and unzip them if not already present.

    Set the environment variable ``LAIR_SKIP_CCG_DOWNLOAD`` to skip the network
    download (e.g. for testing, CI, or read-only/offline environments). When
    skipped and the file is absent, ``lair.background`` will not be importable.
    """
    # Define the path for the CCG filter file
    lair_dir = os.path.dirname(__file__)
    ccg_filter_file = os.path.join(lair_dir, "_ccg_filter.py")

    # Check if the CCG filter file already exists
    if not os.path.exists(ccg_filter_file):
        if os.getenv("LAIR_SKIP_CCG_DOWNLOAD"):
            # Offline/CI/read-only: skip the FTP download (see docstring)
            return

        remote_zf = "user/thoning/ccgcrv/ccg_filter.zip"
        zf = os.path.join(lair_dir, "ccg_filter.zip")

        # Download the zip file from the FTP server
        ftp_download("ftp.gml.noaa.gov", remote_zf, lair_dir)

        # Unzip the downloaded file
        unzip(zf, lair_dir)

        # Cleanup: remove the downloaded zip and unnecessary files
        os.remove(zf)
        os.remove(os.path.join(lair_dir, "ccg_dates.py"))
        os.remove(os.path.join(lair_dir, "ccgcrv.py"))

        # Rename the original file with leading underscore (private module)
        os.rename(os.path.join(lair_dir, "ccg_filter.py"), ccg_filter_file)


setup_ccg_filter()
