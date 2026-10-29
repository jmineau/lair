"""
Tools for code development.

:func:`public_attrs` lists an object's public names.
"""


def public_attrs(obj: object) -> list[str]:
    """
    Return a list of public attributes of an object.

    Parameters
    ----------
    obj : object
        Object to get the public attributes of.

    Returns
    -------
    list[str]
        List of public attributes.

    See Also
    --------
    dir : Lists every name, private ones included.

    Examples
    --------
    >>> from lair.dev import public_attrs
    >>> class Site:
    ...     name = "SLC"
    ...     _cache = {}
    >>> public_attrs(Site())
    ['name']
    """
    return [a for a in dir(obj) if not a.startswith("_")]
