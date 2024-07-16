"""
Init file for UATAQ package.
"""

import datetime as dt
import pandas as pd
from typing import Union, Literal


from lair.utils.clock import TimeRange
from ._laboratory import Laboratory, laboratory, get_site
from .filesystem import DEFAULT_GROUP
from . import instruments, sites


_all_or_mult_strs = Union[Literal['all'], str, list[str], tuple[str, ...], set[str]]

#: UATAQ Laboratory object.
#: 
#: Built from :doc:`UATAQ configuration <config>`.
laboratory: Laboratory


# sites = {SID: laboratory.get_site(SID)  # name conflict
#          for SID in laboratory.sites}  # how much time does this take?


def read_data(SID: str, 
              instruments: _all_or_mult_strs = 'all',
              group: str | None = None, lvl: str | None = None,
              time_range: TimeRange._input_types = None,
              num_processes: int | Literal['max'] = 1,
              file_pattern: str | None = None
              ) -> dict[str, pd.DataFrame]:
    """
    Read data from an instrument at a site.

    Parameters
    ----------
    SID : str
        The site ID.
    instruments : str | list[str] | tuple[str] | set[str] | 'all'
        The instrument(s) to read data from.
    group : str | None
        The group name.
    lvl : str | None
        The data level.
    time_range : str | list[Union[str, dt.datetime, None]] | tuple[Union[str, dt.datetime, None], Union[str, dt.datetime, None]] | slice | None
        The time range to read data. Default is None which reads all available data.
    num_processes : int | 'max'
        The number of processes to use. Default is 1.
    file_pattern : str | None
        A string pattern to filter the file paths.

    Returns
    -------
    dict[str, pd.DataFrame]
        The data.
    """
    site = get_site(SID)
    data = site.read_data(instruments, group, lvl, time_range, num_processes,
                          file_pattern)

    return data


def get_obs(SID: str,
            pollutants: _all_or_mult_strs ='all',
            format: Literal['wide'] | Literal['long'] = 'wide',
            group: str | None = None,
            time_range: TimeRange._input_types = None,
            num_processes: int | Literal['max'] = 1,
            file_pattern: str | None = None) -> pd.DataFrame:
    """
    Get observations from a site.

    Parameters
    ----------
    SID : str
        The site ID.
    pollutants : str | list[str] | tuple[str] | set[str] | 'all'
        The pollutant(s) to get observations for.
    format : 'wide' | 'long'
        The format of the data. Default is 'wide'.
    group : str | None
        The group name.
    time_range : str | list[Union[str, dt.datetime, None]] | tuple[Union[str, dt.datetime, None], Union[str, dt.datetime, None]] | slice | None
        The time range to get observations. Default is None which gets all available data.
    num_processes : int | 'max'
        The number of processes to use. Default is 1.
    file_pattern : str | None
        A string pattern to filter the file paths.

    Returns
    -------
    pd.DataFrame
        The observations.
    """
    site = get_site(SID)
    obs = site.get_obs(pollutants, format, group, time_range, num_processes)

    return obs


def get_recent_obs(SID, recent: str | dt.timedelta = dt.timedelta(days=10),
                   pollutants: _all_or_mult_strs = 'all',
                   format: Literal['wide'] | Literal['long'] = 'wide',
                   group: str | None = None) -> pd.DataFrame:
    """
    Get recent observations from a site.

    Parameters
    ----------
    SID : str
        The site ID.
    recent : str | dt.timedelta
        The recent time range. Default is 10 days.
    pollutants : str | list[str] | tuple[str] | set[str] | 'all'
        The pollutant(s) to get observations for.
    format : 'wide' | 'long'
        The format of the data. Default is 'wide'.
    group : str | None
        The group name.

    Returns
    -------
    pd.DataFrame
        The recent observations.
    """
    site = get_site(SID)
    obs = site.get_recent_obs(recent, pollutants, format, group)

    return obs


__all__ = [
    'sites', 'instruments',
    'laboratory',
    'get_site', 'read_data', 'get_obs', 'get_recent_obs',
    ]
