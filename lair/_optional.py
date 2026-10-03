"""
Centralized optional dependency management.
"""

import importlib


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
        When the module (or a parent package of it) is not installed. Any
        other import error raised while importing an installed module is
        re-raised unchanged.
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        # Only a missing `name` (or a parent package) means the dependency
        # isn't installed. Anything else, e.g. a missing dependency *of*
        # `name`, is a real error and should surface as-is.
        if e.name is None or not (name == e.name or name.startswith(e.name + ".")):
            raise
        raise ImportError(f"Optional `lair` dependency '{name}' not found.") from e
