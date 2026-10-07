"""lair: tools for land-air interactions research."""

import logging
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

import cf_xarray.units  # must be imported before pint_xarray
import pint
import pint_xarray
from pint_xarray import unit_registry as units

# sort_by_dimensionality is private pint API and it moves between releases:
# pint >= 0.26 keeps it in `sorting`, pint <= 0.25 in `_compound_unit_helpers`.
try:
    # pyrefly: ignore[missing-import]
    from pint.delegates.formatter.sorting import sort_by_dimensionality
except ImportError:
    try:
        from pint.delegates.formatter._compound_unit_helpers import (  # pyrefly: ignore[missing-import]
            sort_by_dimensionality,
        )
    except ImportError:
        sort_by_dimensionality = None

from . import config
from .records import ftp_download, unzip

logger = logging.getLogger(__name__)

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


def setup_ccg_filter(lair_dir: str | None = None) -> None:
    """
    Set up the CCG filter module from NOAA GML.

    If ``_ccg_filter.py`` is not already present, download NOAA's
    ``ccg_filter.zip`` into a temporary directory, take ``ccg_filter.py`` from
    it (other zip members are ignored), and move it into place as
    ``_ccg_filter.py`` with an atomic :func:`os.replace`. A failed download or
    unusable zip raises and leaves no partial file behind (at ``import lair``
    the failure is logged as a warning instead, and only ``lair.background``
    is unavailable). Concurrent first
    imports (e.g. a SLURM array) are safe: each installs a complete file
    atomically, and one that finds the file already installed uses it.

    Set the environment variable ``LAIR_SKIP_CCG_DOWNLOAD`` to skip the network
    download (e.g. for testing, CI, or read-only/offline environments). When
    skipped and the file is absent, ``lair.background`` will not be importable.

    Parameters
    ----------
    lair_dir : str, optional
        Directory to install ``_ccg_filter.py`` into. Defaults to the lair
        package directory.
    """
    import tempfile
    import zipfile

    if lair_dir is None:
        lair_dir = os.path.dirname(__file__)
    ccg_filter_file = os.path.join(lair_dir, "_ccg_filter.py")

    if os.path.exists(ccg_filter_file):
        return
    if os.getenv("LAIR_SKIP_CCG_DOWNLOAD"):
        # Offline/CI/read-only: skip the FTP download (see docstring)
        return

    # The temporary directory sits next to the target so os.replace stays on one
    # filesystem (atomic); it is removed on exit, whether or not we succeed.
    with tempfile.TemporaryDirectory(prefix=".ccg_filter-", dir=lair_dir) as tmp:
        ftp_download("ftp.gml.noaa.gov", "user/thoning/ccgcrv/ccg_filter.zip", tmp)
        zf = os.path.join(tmp, "ccg_filter.zip")

        with zipfile.ZipFile(zf) as z:
            members = [
                m for m in z.namelist() if os.path.basename(m) == "ccg_filter.py"
            ]
            if not members:
                raise FileNotFoundError(
                    f"No ccg_filter.py in NOAA's {os.path.basename(zf)}"
                )
            source = z.read(members[0])

        staged = os.path.join(tmp, "_ccg_filter.py")
        with open(staged, "wb") as f:
            f.write(source)

        if os.path.exists(ccg_filter_file):
            # Another process installed it while we were downloading
            return
        os.replace(staged, ccg_filter_file)


def _setup_ccg_filter_or_warn() -> None:
    """
    Install the CCG filter at import; a failure warns instead of raising.

    Only ``lair.background`` needs the filter, so a NOAA outage or a node
    without outbound FTP shouldn't break ``import lair``.
    """
    try:
        setup_ccg_filter()
    except Exception as e:
        logger.warning(
            "Could not install NOAA's CCG filter (%s: %s). lair.background is "
            "unavailable until lair.setup_ccg_filter() succeeds.",
            type(e).__name__,
            e,
        )


_setup_ccg_filter_or_warn()
