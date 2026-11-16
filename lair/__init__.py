"""
lair: tools for land-air interactions research.

lair is a toolkit, one module per topic: import the module you need, as in
``from lair.meteorology import poisson`` (the API reference lists them). The
package itself holds what the modules share: the unit registry
:data:`units` with its :data:`mass_flux` context, and
:func:`setup_ccg_filter`, which fetches NOAA's CCG filter when
:mod:`lair.background` first needs it (``import lair`` downloads nothing). It
also re-exports :mod:`lair.config`, :func:`~lair.records.ftp_download` and
:func:`~lair.records.unzip`, and ``__version__``.

Importing lair changes pint for the whole process: it adds the
:data:`mass_flux` context to the registry and makes pint write a unit's parts
in the order of their dimensions (``kg/km**2/d``).
"""

import hashlib
import importlib
import importlib.util
import logging
import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from pathlib import Path
from types import ModuleType

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
#: See Also
#: --------
#: mass_flux : The context that converts between moles and mass.
#: lair.inventories.convert_units : Converts an array's ``units`` attribute for a pollutant.
#:
#: Examples
#: --------
#: A quantity is a number with a unit, and the registry reads the unit from
#: text, CF spellings such as ``m-2 s-1`` included. ``to`` converts within one
#: dimension; a flux in moles needs the ``mass_flux`` context to become a flux
#: in mass.
#:
#: .. ipython:: python
#:
#:    from lair import units
#:
#:    flux = 10 * units("nmol m-2 s-1")
#:    print(f"{flux:~P}")
#:    print(f"{flux.to('mol km-2 d-1'):~P.3f}")
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
#: See Also
#: --------
#: units : The registry that has the context.
#: lair.inventories.molecular_weight : The molar mass of a pollutant, for ``mw``.
#: lair.inventories.convert_units : Does this conversion for a pollutant's name.
#:
#: Examples
#: --------
#: 10 nmol m-2 s-1 of methane (16.04 g/mol) is 13.86 kg km-2 d-1, and back:
#:
#: .. ipython:: python
#:
#:    from lair import units
#:
#:    mw = 16.04 * units("g/mol")
#:    flux = 10 * units("nmol m-2 s-1")
#:    with units.context("mass_flux", mw=mw):
#:        kg = flux.to("kg km-2 d-1")
#:        mol = kg.to("nmol m-2 s-1")
#:    print(f"{kg:~P.2f}")
#:    print(f"{mol:~P.2f}")
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


#: SHA-256 of NOAA's ``ccg_filter.py`` (the file inside ``ccg_filter.zip``, dated
#: 2024-07-08 in the zip). The zip itself is not hashed: it holds two other
#: files lair does not use, and a repackaged zip would change its hash without
#: changing the code that runs. When NOAA publishes a new filter, this changes.
_CCG_FILTER_SHA256 = "e651b787e438ae74e0f4d13336311ed07a2a876b19dc7213d4650618b39ea6a6"

# The name the filter is imported under; no file by this name exists, so the
# import system finds it only when a stand-in (the docs build's mock) is there.
_CCG_FILTER_MODULE = "lair._noaa_ccg_filter"


def _has_ccg_checksum(path: Path) -> bool:
    """Return whether ``path`` is a file with the expected SHA-256."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() == _CCG_FILTER_SHA256
    except OSError:
        return False


def setup_ccg_filter(directory: str | os.PathLike | None = None) -> Path | None:
    """
    Fetch the CCG filter from NOAA GML into the cache folder.

    Downloads NOAA's ``ccg_filter.zip`` and saves ``ccg_filter.py`` from it,
    unless a copy with the expected SHA-256 is already there. ``lair.background``
    calls this function the first time it is imported (``import lair`` does
    not): :ref:`ccg-filter-download` says what is downloaded and why, what
    happens without a network, and how to skip the download or place the file
    yourself.

    The download is checked against the SHA-256 of the ``ccg_filter.py`` that
    lair was written against before it is saved; a different file raises a
    ``ValueError`` and nothing is kept. A file already in the folder is checked
    the same way before it is used, and replaced by a fresh download when it
    does not match.

    Called on its own, it raises when the download fails, the zip holds no
    ``ccg_filter.py`` or the checksum does not match, and leaves no partial file
    behind. The file is moved into place atomically, so concurrent first uses
    (a Slurm array, for example) are safe: one that finds the file already
    installed uses it.

    Parameters
    ----------
    directory : str or os.PathLike, optional
        Folder to save ``ccg_filter.py`` into, made if needed. Defaults to the
        ``ccg_filter`` folder inside :func:`lair.config.get_cache_dir`.

    Returns
    -------
    pathlib.Path or None
        The path of ``ccg_filter.py``, or None when it is not there and
        ``LAIR_SKIP_CCG_DOWNLOAD`` forbids the download.

    Raises
    ------
    ValueError
        If the file NOAA serves does not have the expected checksum.

    See Also
    --------
    lair.background.thoning : Fits NOAA's curve, which needs the filter.
    lair.config.get_cache_dir : The cache folder.

    Examples
    --------
    >>> import lair
    >>> lair.setup_ccg_filter()  # doctest: +SKIP
    PosixPath('/home/me/.cache/lair/ccg_filter/ccg_filter.py')
    """
    import tempfile
    import zipfile

    if directory is None:
        directory = config.get_cache_dir() / "ccg_filter"
    ccg_filter_file = Path(directory) / "ccg_filter.py"

    if _has_ccg_checksum(ccg_filter_file):
        return ccg_filter_file
    if os.getenv("LAIR_SKIP_CCG_DOWNLOAD"):
        # Offline/CI/read-only: skip the FTP download (see docstring)
        if ccg_filter_file.exists():
            logger.warning(
                "Not using %s: its checksum is not NOAA's current filter, and "
                "LAIR_SKIP_CCG_DOWNLOAD forbids fetching a new copy.",
                ccg_filter_file,
            )
        return None
    if ccg_filter_file.exists():
        logger.warning("Replacing %s: its checksum does not match.", ccg_filter_file)

    # The temporary directory sits next to the target so os.replace stays on one
    # filesystem (atomic); it is removed on exit, whether or not we succeed.
    ccg_filter_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".ccg_filter-", dir=ccg_filter_file.parent
    ) as tmp:
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

        digest = hashlib.sha256(source).hexdigest()
        if digest != _CCG_FILTER_SHA256:
            raise ValueError(
                "The ccg_filter.py that NOAA GML serves has SHA-256 "
                f"{digest}, not the {_CCG_FILTER_SHA256} this version of lair "
                "was written against, so lair will not use it. NOAA may have "
                "published a new filter (check "
                "https://gml.noaa.gov/ccgg/mbl/crvfit/crvfit.html and open an "
                "issue at https://github.com/jmineau/lair/issues), or the "
                "download was altered."
            )

        staged = os.path.join(tmp, "ccg_filter.py")
        with open(staged, "wb") as f:
            f.write(source)

        if _has_ccg_checksum(ccg_filter_file):
            # Another process installed it while we were downloading
            return ccg_filter_file
        os.replace(staged, ccg_filter_file)
    return ccg_filter_file


def _load_ccg_filter() -> ModuleType:
    """
    Return the CCG filter module, fetching it into the cache folder if needed.

    What ``lair.background`` imports ``ccgFilter`` from. A failure to get the
    file (skipped, no network, bad checksum) raises an ``ImportError`` that
    names :func:`setup_ccg_filter`; an error inside NOAA's code, such as a missing
    ``scipy``, is raised as it is.
    """
    try:
        return importlib.import_module(_CCG_FILTER_MODULE)
    except ModuleNotFoundError as e:
        if e.name != _CCG_FILTER_MODULE:
            raise

    try:
        path = setup_ccg_filter()
    except Exception as e:
        raise ImportError(
            "lair.background needs NOAA's CCG filter, which could not be set up "
            f"({type(e).__name__}: {e}). Run lair.setup_ccg_filter() once the "
            "cause is fixed (it needs outbound FTP to ftp.gml.noaa.gov)."
        ) from e
    if path is None:
        raise ImportError(
            "lair.background needs NOAA's CCG filter, which isn't installed, "
            "and LAIR_SKIP_CCG_DOWNLOAD is set, so lair did not download it. "
            "Unset the variable and run lair.setup_ccg_filter()."
        )

    spec = importlib.util.spec_from_file_location(_CCG_FILTER_MODULE, path)
    if spec is None or spec.loader is None:  # not a .py file we can load
        raise ImportError(f"Cannot load NOAA's CCG filter from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_CCG_FILTER_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[_CCG_FILTER_MODULE]
        raise
    return module
