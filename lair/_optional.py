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
    return importlib.import_module(name)

