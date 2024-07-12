"""
lair.utils.clock
~~~~~~~~~~~~~~~~

This module provides utility functions for working with time and dates.
"""

from contextlib import ContextDecorator
from dataclasses import dataclass, field
import datetime as dt
from functools import partial
import pandas as pd
import re
import time
from typing import Any, Callable, ClassVar, Dict, Literal, Optional, Union
from zoneinfo import ZoneInfo

AFTERNOON = [12, 13, 14, 15, 16] # HH Local Standard Time
SEASONS = {1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA',
           7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON', 12: 'DJF'}


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


@dataclass
class Timer(ContextDecorator):
    """
    Time your code using a class, context manager, or decorator
    
    https://realpython.com/python-timer
    """

    timers: ClassVar[Dict[str, float]] = {}
    name: str | None = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

    def __post_init__(self) -> None:
        """Initialization: add timer to dict of timers"""
        if self.name:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise Timer.TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise Timer.TimerError("Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time

    def reset_timers(self):
        """Reset class timers"""
        Timer.timers = {}

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.stop()


def datetime_accessor(obj, accessor='dt'):
    """
    Returns the datetime accessor of the object.
    """
    if hasattr(obj, accessor):
        return getattr(obj, accessor)
    return obj


# ----- Time Aggregation ----- #

def diurnal(data: pd.DataFrame, freq: str='1H', statistic: str='mean',
            method: Literal['floor', 'ceil', 'round']='floor'):
    """
    Aggregate the data to the specified frequency and compute the statistic for each group.

    Parameters
    ----------
    data : pd.DataFrame
        The data to aggregate. Panda DataFrame with a DateTimeIndex.
    freq : str
        The frequency to aggregate the data to. Sub-Daily.
    statistic : str or list of str, optional
        The statistic to compute for each group. Default is 'mean'.
    method : one of 'floor', 'ceil', 'round'
        The method to use to round the index to the nearest freq.

    Returns
    -------
    pd.DataFrame
        The aggregated data.
    """
    resolution = getattr(data.index, method)(freq).time
    agg = data.groupby(resolution).agg(statistic)

    return agg


def seasonal(data: pd.DataFrame, statistic: str='mean') -> pd.DataFrame:
    # Resample the data to the start of quarters and group by year
    df = data.resample('QS-DEC').agg(statistic)
    df['season'] = df.index.month.map(SEASONS)

    # doesnt actually take the mean, just regroups them into season:year
    df = df.set_index(['season', df.index.year]) 

    return df


# ----- Time Conversion ----- #

def seconds_in_year(year: int) -> float:
    """
    Calculate the number of seconds in a year.

    Parameters
    ----------
    year : int
        The year to calculate the number of seconds for.

    Returns
    -------
    float
        The number of seconds in the year.
    """
    this_year = dt.datetime(year, 1, 1)
    next_year = dt.datetime(year + 1, 1, 1)
    return (next_year - this_year).total_seconds()


def dt2decimalDate(datetime: dt.datetime) -> float:
    """
    Convert a datetime object to a decimal date.

    Parameters
    ----------
    datetime : dt.datetime
        The datetime object to convert.

    Returns
    -------
    float
        The decimal date.
    """
    this_year = dt.datetime(datetime.year, 1, 1)
    total_seconds = (datetime - this_year).total_seconds()
    total_seconds_year = seconds_in_year(datetime.year)
    return datetime.year + (total_seconds / total_seconds_year)


def decimalDate2dt(decimalDate: float) -> dt.datetime:
    """
    Convert a decimal date to a datetime object.

    Parameters
    ----------
    decimalDate : float
        The decimal date to convert.

    Returns
    -------
    dt.datetime
        The datetime object.
    """
    year = int(decimalDate)
    rem = decimalDate - year

    this_year = dt.datetime(year, 1, 1)
    total_seconds_year = seconds_in_year(year)
    return this_year + dt.timedelta(seconds=total_seconds_year * rem)


# ----- Time Zones ----- #

def convert_timezones(x, totz, fromtz=None, localize=False,
                      driver=None):
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

    if driver is None:
        times = x
        # If the times are not tz-aware, assign them the fromtz timezone.
        if fromtz is not None and not any(t.tzinfo for t in times):
            fromtz = ZoneInfo(fromtz)
            times = [t.replace(tzinfo=fromtz) for t in times]

        # Convert the times to the specified timezone.
        totz = ZoneInfo(totz)
        converted_times = [t.astimezone(totz) for t in times]

        # If localize is True, localize the times to the totz timezone.
        if localize:
            converted_times = [t.replace(tzinfo=None) for t in converted_times]

        return converted_times
    elif driver == 'pandas':
        data = x.copy(deep=True)

        # If x is a DataFrame, convert DateTimeIndex
        # If x is a Series, convert the series
        if isinstance(data, pd.DataFrame):
            times = data.index
            index = True
        elif isinstance(data, pd.Series):
            times = data
            index = False
        else:
            raise ValueError("x is not a DataFrame or Series")

        # If the times are not tz-aware, assign them the fromtz timezone.
        if fromtz and datetime_accessor(times).tz is None:
            times = datetime_accessor(times).tz_localize(fromtz) 

        # Convert the times to the specified timezone.
        times = datetime_accessor(times).tz_convert(totz)

        # If localize is True, localize the times to the totz timezone.
        if localize:
            times = datetime_accessor(times).tz_localize(None)

        if index:
            data.index = times
        else:
            data = times
        return data
    else:
        raise ValueError("Invalid driver")


UTC2 = partial(convert_timezones, fromtz='UTC')
UTC2MST = partial(UTC2, totz='MST')
UTC2MTN = partial(UTC2, totz='America/Denver')
MST2UTC = partial(convert_timezones, fromtz='MST', totz='UTC')
MTN2UTC = partial(convert_timezones, fromtz='America/Denver', totz='UTC')
