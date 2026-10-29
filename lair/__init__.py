"""
lair: tools for land-air interactions research.

lair is a toolkit, one module per topic: import the module you need, as in
``from lair.meteorology import poisson`` (the API reference lists them). The
package itself holds what the modules share: the unit registry
:data:`units` with its :data:`mass_flux` context, and
:func:`setup_ccg_filter`, which ``import lair`` calls. It also re-exports
:mod:`lair.config`, :func:`~lair.records.ftp_download` and
:func:`~lair.records.unzip`, and ``__version__``.

Importing lair changes pint for the whole process: it adds the
:data:`mass_flux` context to the registry and makes pint write a unit's parts
in the order of their dimensions (``kg/km**2/d``).
"""

import logging
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

import cf_xarray.units  # must be imported before pint_xarray
import pint
import pint_xarray

# sort_by_dimensionality prints a unit's parts in dimension order (kg/km**2/d,
# where pint's default gives kg/d/km**2). pint >= 0.26 exports it; pint 0.24 and
# 0.25, the newest pint on Python 3.11, have it only in a private module. When
# neither has it (pint older than 0.24, or a pint that moves it again) the import
# fails here, on purpose: without it Inventory.get_units mislabels a flux unit.
# pyrefly checks against the installed pint, so it reports whichever import that
# pint lacks: each suppression below is for one pint version.
try:
    # pyrefly: ignore[missing-module-attribute]
    from pint.delegates.formatter import sort_by_dimensionality
except ImportError:
    from pint.delegates.formatter._compound_unit_helpers import (
        # pyrefly: ignore[missing-module-attribute]
        sort_by_dimensionality,
    )

from . import config
from .records import ftp_download, unzip

#: The pint unit registry lair uses, which is pint-xarray's, so a quantified
#: xarray object and a plain quantity share it. It has the :data:`mass_flux`
#: context.
#: :doc:`/examples/units` converts a methane flux with it.
#:
#: :meta hide-value:
units = pint_xarray.unit_registry

logger = logging.getLogger(__name__)

try:
    __version__ = _version("lair")  # set by setuptools-scm from git tags
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0+unknown"


def _molar_mass(mw):
    """Return ``mw`` if it is a molar mass, else raise a TypeError that names it."""
    if not isinstance(mw, pint.Quantity) or not mw.check("[mass] / [substance]"):
        raise TypeError(
            "The mass_flux context needs mw, the molar mass as a quantity, "
            "e.g. units.context('mass_flux', mw=16.04 * units('g/mol')); "
            f"got mw={mw!r}"
        )
    return mw


#: The pint context that converts a flux between moles and mass, given ``mw``.
#:
#: It converts ``[substance]/[area]/[time]`` to ``[mass]/[area]/[time]`` and
#: back, multiplying or dividing by ``mw``, the molar mass as a quantity in
#: g/mol: ``units.context("mass_flux", mw=16.04 * units("g/mol"))``. Without
#: ``mw``, or with a plain number, the conversion raises a ``TypeError`` that
#: names ``mw``. Importing lair adds it to :data:`units`; :doc:`/examples/units`
#: shows it on a methane flux.
#:
#: :meta hide-value:
mass_flux = pint.Context("mass_flux")
mass_flux.add_transformation(
    "[substance] / [area] / [time]",
    "[mass] / [area] / [time]",
    # pint types the callback narrowly; (units, value, mw) is what it calls
    # pyrefly: ignore[bad-argument-type]
    lambda units, substance, mw=None: substance * _molar_mass(mw),
)
mass_flux.add_transformation(
    "[mass] / [area] / [time]",
    "[substance] / [area] / [time]",
    # pyrefly: ignore[bad-argument-type]
    lambda units, mass, mw=None: mass / _molar_mass(mw),
)
units.add_context(mass_flux)

# Set default pint sorting function to sort by dimensionality.
# Not cosmetic: inventories.Inventory.get_units splits a flux unit on "/"
# and reads its parts in this order.
units.formatter.default_sort_func = sort_by_dimensionality


def setup_ccg_filter(lair_dir: str | None = None) -> None:
    """
    Set up the CCG filter module from NOAA GML.

    Installs ``_ccg_filter.py`` into ``lair_dir``, from NOAA's ``ccg_filter.zip``,
    unless the file is already there. ``import lair`` calls this function:
    :ref:`first-import-download` says what is downloaded and why, what happens
    without a network, and how to skip the download or place the file yourself.

    Called on its own, it raises when the download fails or the zip holds no
    ``ccg_filter.py``, and leaves no partial file behind. The file is moved into
    place atomically, so concurrent first imports (a Slurm array, for example) are
    safe: one that finds the file already installed uses it.

    Parameters
    ----------
    lair_dir : str, optional
        Directory to install ``_ccg_filter.py`` into. Defaults to the lair
        package directory.

    See Also
    --------
    lair.background.thoning : Fits NOAA's curve, which needs the filter.

    Examples
    --------
    >>> import lair
    >>> lair.setup_ccg_filter()  # doctest: +SKIP
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
