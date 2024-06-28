"""
lair.uataq.filesystem.__init__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

UATAQ filesystem init file.
"""

from ._filesystem import (DataFile, GroupSpace,
                          groups, lvls,
                          filter_datafiles, parse_datafiles)
from . import groupspaces


DEFAULT_GROUP = 'lin'


def get_group(group: str | None) -> str:
    """
    Get the group name.

    Parameters
    ----------
    group : str
        The group name.

    Returns
    -------
    str
        The group name.
    """
    if group is None:
        return DEFAULT_GROUP
    elif group not in groups:
        raise ValueError(f'Invalid group: {group}')
    else:
        return group