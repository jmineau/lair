"""
Where lair finds data and keeps its cache.

lair has no built-in data paths. A reader takes its folder as an argument and
otherwise reads a ``LAIR_*`` environment variable through
:func:`get_data_dir`; :func:`get_cache_dir` gives the cache folder. Both read
the environment when called, not at import. :ref:`data-locations` lists the
variables.
"""

import os
from pathlib import Path

###############
# Directories #
###############

# lair has no built-in data locations. Readers of a data archive take an
# explicit directory argument and otherwise fall back to a LAIR_* environment
# variable (see `get_data_dir`), e.g. LAIR_INVENTORY_DIR, LAIR_SOUNDING_DIR,
# LAIR_CARBONTRACKER_DIR, LAIR_GML_DIR. LAIR_CACHE_DIR has a default (see
# `get_cache_dir`), so it never raises.


def get_data_dir(env_var: str, path: str | os.PathLike | None = None) -> Path:
    """
    Resolve a data directory.

    Parameters
    ----------
    env_var : str
        Environment variable to fall back to, e.g. ``'LAIR_INVENTORY_DIR'``.
    path : str | os.PathLike, optional
        Explicit directory. Takes precedence over the environment variable.

    Returns
    -------
    Path
        The data directory.

    Raises
    ------
    ValueError
        If neither ``path`` nor the environment variable is set.

    See Also
    --------
    get_cache_dir : The cache folder, which has a default.

    Examples
    --------
    An explicit folder wins over the variable:

    >>> import os
    >>> from lair.config import get_data_dir
    >>> os.environ["LAIR_INVENTORY_DIR"] = "/data/inventories"
    >>> get_data_dir("LAIR_INVENTORY_DIR").as_posix()
    '/data/inventories'
    >>> get_data_dir("LAIR_INVENTORY_DIR", "/elsewhere").as_posix()
    '/elsewhere'

    With neither, the error names the variable:

    >>> del os.environ["LAIR_INVENTORY_DIR"]
    >>> get_data_dir("LAIR_INVENTORY_DIR")
    Traceback (most recent call last):
        ...
    ValueError: No data directory given: pass it explicitly or set the LAIR_INVENTORY_DIR environment variable.
    """
    if path is None:
        path = os.environ.get(env_var)
    if not path:
        raise ValueError(
            f"No data directory given: pass it explicitly or set "
            f"the {env_var} environment variable."
        )
    return Path(path)


def get_cache_dir() -> Path:
    """
    Resolve the cache directory.

    The environment variable ``LAIR_CACHE_DIR`` is read each time this is
    called, not when lair is imported, so setting it after ``import lair``
    works. The folder is not made here: the code that caches makes it the
    first time it writes there, so ``import lair`` works where the home
    directory is read-only.

    Returns
    -------
    Path
        ``$LAIR_CACHE_DIR``, or ``.cache/lair`` in the home directory when
        the variable is unset or empty.

    See Also
    --------
    get_data_dir : The folder of a data archive, which has no default.
    lair.records.Cacher : Writes the cache files.

    Examples
    --------
    >>> import os
    >>> from lair.config import get_cache_dir
    >>> os.environ["LAIR_CACHE_DIR"] = "/scratch/lair_cache"
    >>> get_cache_dir().as_posix()
    '/scratch/lair_cache'
    >>> del os.environ["LAIR_CACHE_DIR"]
    >>> get_cache_dir().parts[-2:]
    ('.cache', 'lair')
    """
    path = os.environ.get("LAIR_CACHE_DIR")
    if path:
        return Path(path)
    return Path(os.path.expanduser("~"), ".cache", "lair")


# LAIR
LAIR_DIR = os.path.dirname(__file__)
