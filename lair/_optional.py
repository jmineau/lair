"""
Centralized optional dependency management.
"""

import importlib

# The `lair` extra that installs each optional package, keyed by the name
# `import_optional_dependency` is given (the import name, which is also the
# package name for all of these). `tests/optional` checks this table against
# `[project.optional-dependencies]` in `pyproject.toml` and that every call in
# `lair/` names a package in it, so it cannot drift silently. A package in more
# than one extra is listed under the smallest (the fewest packages to install);
# `complete`, which holds every extra, is never named.
_EXTRA_OF = {
    "boto3": "requests",
    "cartopy": "geo",
    "fastkml": "formats",
    "lxml": "formats",
    "molmass": "science",
    "networkx": "geo",
    "numcodecs": "formats",
    "pyproj": "geo",
    "rasterio": "geo",
    "requests": "requests",
    "rioxarray": "geo",
    "s3fs": "requests",
    "scipy": "science",
    "shapely": "geo",
    "siphon": "requests",
    "xesmf": "regridding",
    "zarr": "formats",
}

# lair is installed from GitHub: the `lair` on PyPI is an unrelated project.
_SOURCE = "git+https://github.com/jmineau/lair.git"
_INSTALL_DOCS = "https://jmineau.github.io/lair/#installation"


def import_optional_dependency(name: str):
    """
    Import an optional dependency.

    Parameters
    ----------
    name : str
        The module name to import.

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    ImportError
        When the module (or a parent package of it) is not installed. The
        message names the `lair` extra that provides it and the command that
        installs the extra, for example ``pip install "lair[geo] @
        git+https://github.com/jmineau/lair.git"``. Any other import error
        raised while importing an installed module is re-raised unchanged.
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        # Only a missing `name` (or a parent package) means the dependency
        # isn't installed. Anything else, e.g. a missing dependency *of*
        # `name`, is a real error and should surface as-is.
        if e.name is None or not (name == e.name or name.startswith(e.name + ".")):
            raise
        raise ImportError(_not_found_message(name)) from e


def _not_found_message(name: str) -> str:
    """Say which extra provides `name` and how to install it."""
    message = f"Optional `lair` dependency '{name}' not found."
    extra = _EXTRA_OF.get(name.split(".")[0])
    if extra is None:
        return message
    return (
        f"{message} It is in the `{extra}` extra. Install it with "
        f'`pip install "lair[{extra}] @ {_SOURCE}"` '
        f"(`uv pip install` in a uv environment). See {_INSTALL_DOCS}"
    )
