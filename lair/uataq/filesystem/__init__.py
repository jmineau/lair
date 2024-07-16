"""
UATAQ filesystem init file.

Provides access to the UATAQ filesystem through the use of `DataFile`
and `GroupSpace` objects.
"""

from ._filesystem import (DataFile, GroupSpace,
                          groups, lvls,
                          filter_datafiles, parse_datafiles)
from . import groupspaces

#: Default group to read data from.
DEFAULT_GROUP: str = 'lin'
# Update lineno in docs if this changes.

#: Groups dictionary to store GroupSpace objects.
groups: dict

#: Levels of data processing.
lvls: dict


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


__all__ = ['groupspaces',
           'DataFile', 'GroupSpace',
           'groups', 'lvls',
           'filter_datafiles', 'parse_datafiles']
