"""
Small helpers used across lair.

:func:`updating_print` writes a progress message over the current line, and
:class:`DotDict` is a dictionary whose keys are also attributes.
"""


def updating_print(msg):
    """
    Print *msg* over the current line (a carriage return, no newline).

    See Also
    --------
    print : Starts a new line each time.

    Examples
    --------
    Each call returns to the start of the line, so a terminal shows only the
    last message:

    >>> from lair.utils import updating_print
    >>> for i in range(3):
    ...     updating_print(f"file {i + 1} of 3")  # doctest: +SKIP
    """
    print(f"\r{msg}", end="")


class DotDict(dict):
    """
    A dictionary whose keys are also attributes (``d.key`` as well as ``d["key"]``).

    A dictionary stored in a ``DotDict`` is wrapped when it is first read, so
    ``d.a.b`` works at any depth, and ``d.a.b = 2`` changes ``d``.

    See Also
    --------
    dict : What it extends.

    Examples
    --------
    >>> from lair.utils import DotDict
    >>> d = DotDict({"site": {"name": "SLC", "elevation": 1289}})
    >>> d.site.name
    'SLC'
    >>> d.site.elevation = 1300
    >>> d["site"]["elevation"]
    1300
    >>> d.missing
    Traceback (most recent call last):
        ...
    AttributeError: missing
    """

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError:
            # AttributeError keeps hasattr, getattr(..., default), copy working
            raise AttributeError(key) from None
        if type(val) is dict:
            # Store the wrapped dict back so writes like `d.a.b = 2` stick
            val = DotDict(val)
            self[key] = val
        return val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        return dir(dict) + list(self.keys())
