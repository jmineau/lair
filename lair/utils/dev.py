"""
lair.utils.dev

Module of tools for code development.
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
    """
    return [a for a in dir(obj) if not a.startswith('_')]
