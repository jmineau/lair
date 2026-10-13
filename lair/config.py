"""
Config module for lair package
"""

import os
from pathlib import Path

###############
# Directories #
###############

# lair has no built-in data locations. Readers of a data archive take an
# explicit directory argument and otherwise fall back to a LAIR_* environment
# variable (see `get_data_dir`), e.g. LAIR_INVENTORY_DIR, LAIR_SOUNDING_DIR,
# LAIR_CARBONTRACKER_DIR, LAIR_GML_DIR.


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
    """
    if path is None:
        path = os.environ.get(env_var)
    if not path:
        raise ValueError(
            f"No data directory given: pass it explicitly or set "
            f"the {env_var} environment variable."
        )
    return Path(path)


# LAIR
LAIR_DIR = os.path.dirname(__file__)

#: User Cache Directory. Importing lair does not create it: the code that
#: caches (``lair.noaa``) makes the folder the first time it writes there, so
#: ``import lair`` works where the home directory is read-only.
CACHE_DIR = os.getenv(
    "LAIR_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "lair")
)
