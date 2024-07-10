"""
lair.utils.clock
~~~~~~~~~~~~~~~~

This module provides utility functions for working with time and dates.
"""

from collections import namedtuple
import datetime as dt
from functools import partial
import pandas as pd
import re
from typing import Union
import pytz

AFTERNOON = range(12, 17)  # Local Standard Time
AFTERNOONUTC = range(18, 23)  # AFTERNOON UTC for SLC

SEASONS = {1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA',
           7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON', 12: 'DJF'}


def get_afternoons(times, hours=AFTERNOON):
    afternoons = times[times.hour.isin(hours)]

    return afternoons


def convert_timezones(times, fromtz, totz, localize=False):
    """
    Convert the times from one timezone to another.

    Args:
        times (list): A list of datetime objects.
        fromtz (str): The timezone of the input times.
        totz (str): The timezone to convert the times to.
        localize (bool): If True, the times will be localized to the totz timezone.

    Returns:
        list: A list of datetime objects in the totz timezone.
    """
    # If the times are not tz-aware, assign them the fromtz timezone.
    if not any(t.tzinfo for t in times):
        times = [t.replace(tzinfo=pytz.timezone(fromtz)) for t in times]

    # Convert the times to the specified timezone.
    converted_times = [t.astimezone(pytz.timezone(totz)) for t in times]

    # If localize is True, localize the times to the totz timezone.
    if localize:
        converted_times = [t.replace(tzinfo=None) for t in converted_times]

    return converted_times


UTC2 = partial(convert_timezones, fromtz='UTC')
UTC2MST = partial(UTC2, totz='MST')
UTC2MTN = partial(UTC2, totz='America/Denver')
MST2UTC = partial(convert_timezones, fromtz='MST', totz='UTC')
MTN2UTC = partial(convert_timezones, fromtz='America/Denver', totz='UTC')


def dt2decimalDate(datetime):
    # https://stackoverflow.com/a/36949905
    start = dt.date(datetime.year, 1, 1).toordinal()
    year_length = dt.date(datetime.year+1, 1, 1).toordinal() - start
    return datetime.year + float(datetime.toordinal() - start) / year_length


def decimalDate2dt(decimalDate):
    # https://stackoverflow.com/a/20911144
    year = int(decimalDate)
    rem = decimalDate - year

    base = dt.datetime(year, 1, 1)
    datetime = base + dt.timedelta(seconds=(base.replace(year=base.year + 1)
                                            - base).total_seconds() * rem)

    return datetime


class TimeRange:
    _input_types = Union[str,
                        list[Union[str, dt.datetime, None]],
                        tuple[Union[str, dt.datetime, None],
                              Union[str, dt.datetime, None]],
                        slice,
                        None]
    """
    TimeRange class to represent a time range with start and stop times.

    Attributes
    ----------
    start : dt.datetime | None
        The start time of the time range.
    stop : dt.datetime | None
        The stop time of the time range.

    Methods
    -------
    parse_iso(string: str, inclusive: bool = False) -> dt.datetime
        Parse the ISO8601 formatted time string and return a datetime object.
    """

    def __init__(self, time_range: 'TimeRange' | _input_types = None,
                 start: Union[_input_types, dt.datetime] = None,
                 stop: Union[_input_types, dt.datetime] = None):
        """
        Initialize a TimeRange object with the specified time range.

        Parameters
        ----------
        time_range : str | list[Union[str, dt.datetime, None]] | tuple[Union[str, dt.datetime, None], Union[str, dt.datetime, None]] | slice | None
            _description_, by default None
        start : Union[_input_types, dt.datetime], optional
            _description_, by default None
        stop : Union[_input_types, dt.datetime], optional
            _description_, by default None

        Raises
        ------
        ValueError
            _description_
        """
        assert not all([time_range, any([start, stop])]), "Cannot specify both time_range and start/stop"

        self._start = None
        self._stop = None

        if isinstance(time_range, TimeRange):
            start, stop = time_range.start, time_range.stop
        elif not any([time_range, start, stop]):
            start = None
            stop = None
        elif time_range:
            if isinstance(time_range, str):
                # Handle the case when time_range is a string
                start = TimeRange.parse_iso(time_range)
                stop = TimeRange.parse_iso(time_range, inclusive=True)
            elif isinstance(time_range, (list, tuple)) and len(time_range) == 2:
                # Handle the case when time_range is a list/tuple
                #   representing start and stop
                start, stop = time_range
            elif isinstance(time_range, slice):
                # Handle the case when time_range is a slice
                start, stop = time_range.start, time_range.stop
            else:
                raise ValueError("Invalid time_range format")

        self.start = start
        self.stop = stop

    def __repr__(self):
        return f"TimeRange(start={self.start}, stop={self.stop})"

    def __str__(self):
        if not any([self.start, self.stop]):
            return 'Entire Observation Period'
        elif not self.start:
            return f'Before {self.stop}'
        elif not self.stop:
            return f'After {self.start}'
        else:
            return f'{self.start} to {self.stop}'

    def __iter__(self):
        return iter([self.start, self.stop])

    @property
    def start(self) -> dt.datetime | None:
        return self._start

    @start.setter
    def start(self, start):
        if start is None or isinstance(start, dt.datetime):
            self._start = start
        elif isinstance(start, str):
            self._start = TimeRange.parse_iso(start)
        else:
            raise ValueError("Invalid start time format")

    @property
    def stop(self) -> dt.datetime | None:
        return self._stop

    @stop.setter
    def stop(self, stop):
        if stop is None or isinstance(stop, dt.datetime):
            self._stop = stop
        elif isinstance(stop, str):
            self._stop = TimeRange.parse_iso(stop, inclusive=True)
        else:
            raise ValueError("Invalid stop time format")

    @staticmethod
    def parse_iso(string: str, inclusive: bool = False) -> dt.datetime:
        """
        Parse the ISO8601 formatted time string and return a namedtuple with the parsed components.

        Parameters
        ----------
        string : str
            The ISO8601 formatted time string.
        inclusive

        Returns
        -------
        dt.datetime
            The parsed datetime object.

        Raises
        ------
        ValueError
            If the time_str format is invalid.
        """
        # Parse time_range string using regex assuming ISO8601 format
        iso8601 = (r'^(?P<year>\d{4})-?(?P<month>\d{2})?-?(?P<day>\d{2})?'
                   r'[T\s]?(?P<hour>\d{1,2})?:?(?:\d{2})?')
        match = re.match(iso8601, string)
        if not match:
            raise ValueError("Invalid time string format")

        components = match.groupdict()
        year = int(components['year'])
        month = int(components['month'] or 1)
        day = int(components['day'] or 1)
        hour = int(components['hour'] or 0)

        start = dt.datetime(year, month, day, hour)

        # Determine the stop time based on the inclusive flag
        if inclusive:
            if components['year'] and not components['month']:
                'YYYY'
                stop = dt.datetime(year + 1, 1, 1)
            elif components['month'] and not components['day']:
                'YYYY-MM'
                mm = month + 1 if month < 12 else 1
                yyyy = year + 1 if month == 12 else year
                stop = dt.datetime(yyyy, mm, 1)
            elif components['day'] and not components['hour']:
                'YYYY-MM-DD'
                stop = start + dt.timedelta(days=1)
            elif components['hour']:
                'YYYY-MM-DDTHH'
                stop = start + dt.timedelta(hours=1)
            else:
                raise ValueError("Invalid time string format")

            return stop
        else:
            return start



def diurnal(data, statistic='mean', freq='1H'):
    resolution = data.index.round(freq).time
    agg = data.groupby(resolution).agg(statistic)

    return agg


def seasonal(data: pd.DataFrame, statistic: str='mean') -> pd.DataFrame:
    # Resample the data to the start of quarters and group by year
    df = data.resample('QS-DEC').agg(statistic)
    df['season'] = df.index.month.map(SEASONS)

    # doesnt actually take the mean, just regroups them into season:year
    df = df.set_index(['season', df.index.year]) 

    return df