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
        When the module is not found.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        raise ImportError(f"Optional `lair` dependency '{name}' not found.")