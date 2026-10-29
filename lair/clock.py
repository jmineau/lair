"""
Times, time zones and time ranges.

:func:`convert_timezones` moves times between zones (a time without a zone
needs ``fromtz``), and :func:`UTC2MTN` and its siblings are its common cases.
:class:`TimeRange` is a start and an end parsed from ISO strings; the
remaining functions turn times into decimal years (:func:`dt2decimalDate`),
intervals, and diurnal and seasonal aggregates.
:doc:`/examples/clock` shows the time-zone rules and the afternoon mean.
"""

import datetime as dt
import re
import time
from collections.abc import Callable, Sequence
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Union
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = [
    "AFTERNOON",
    "SEASONS",
    "TimeRange",
    "Timer",
    "datetime_accessor",
    "regular_times_to_intervals",
    "diurnal",
    "seasonal",
    "dt2decimalDate",
    "decimalDate2dt",
    "convert_timezones",
    "UTC2",
    "UTC2MST",
    "UTC2MTN",
    "MST2UTC",
    "MTN2UTC",
]

#: The afternoon hours, 12 to 16 local standard time, when the boundary layer is
#: deep and a site sees regional air.
#:
#: Put the data in local time first, then ``index.hour.isin(AFTERNOON)`` selects
#: them (the :doc:`clock example </examples/clock>` shows it).
#:
#: .. seealso:: :func:`convert_timezones`, which puts the data in local time.
AFTERNOON = [12, 13, 14, 15, 16]

#: The meteorological season of each month (1 to 12), as ``'DJF'``, ``'MAM'``,
#: ``'JJA'`` or ``'SON'``.
#:
#: :func:`seasonal` labels its groups with these.
#:
#: .. seealso:: :func:`seasonal`, which groups data by these seasons.
SEASONS = {
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
    12: "DJF",
}


class TimeRange:
    """
    TimeRange class to represent a time range with start and stop times.

    The range includes its start and not its stop, so ``TimeRange("2024")``
    covers 2024 and stops at 2025-01-01 00:00.

    Parameters
    ----------
    time_range : TimeRange | str | list | tuple | slice | None, optional
        The whole range in one argument: another ``TimeRange``; an ISO 8601
        string, which covers the period it names (``"2024"`` is all of
        2024, ``"2024-07-04"`` that day); a ``(start, stop)`` pair; or a
        slice. With no ``start`` or ``stop`` either, the default (None) is a
        range open at both ends.
    start : str | dt.datetime | numpy.datetime64 | None, optional
        The start, when ``time_range`` is not given. A string stands for
        the start of the period it names. None leaves the range open at the
        start.
    stop : str | dt.datetime | numpy.datetime64 | None, optional
        The stop, when ``time_range`` is not given. It is not part of the
        range. A string stands for the end of the period it names:
        ``"2024"`` stops at 2025-01-01 00:00. None leaves the range open at
        the end.

    Raises
    ------
    ValueError
        If ``time_range`` is given together with ``start`` or ``stop``, or
        if any of them has an unrecognized format.

    Attributes
    ----------
    start : dt.datetime | None
        The start time of the time range.
    stop : dt.datetime | None
        The stop time of the time range.
    total_seconds : float
        The total number of seconds in the time range.

    Methods
    -------
    parse_iso(string: str, inclusive: bool = False) -> dt.datetime
        Parse the ISO8601 formatted time string and return a datetime object.

    See Also
    --------
    TimeRange.parse_iso : How a string becomes a start or a stop.

    Examples
    --------
    >>> import datetime as dt
    >>> from lair.clock import TimeRange
    >>> july = TimeRange("2024-07")
    >>> july.start, july.stop
    (datetime.datetime(2024, 7, 1, 0, 0), datetime.datetime(2024, 8, 1, 0, 0))
    >>> dt.datetime(2024, 7, 31, 23) in july
    True
    >>> TimeRange(start="2024-07-04")
    TimeRange(start=2024-07-04 00:00:00, stop=None)
    >>> str(TimeRange())
    'Entire Observation Period'
    """

    # typing.Union, not X | Y: the annotation below adds the "TimeRange" forward
    # reference to it at runtime, which only typing.Union accepts.
    _input_types = Union[  # noqa: UP007
        str,
        list[str | dt.datetime | None],
        tuple[str | dt.datetime | None, str | dt.datetime | None],
        slice,
        None,
    ]

    def __init__(
        self,
        time_range: "TimeRange" | _input_types = None,
        start: _input_types | dt.datetime = None,
        stop: _input_types | dt.datetime = None,
    ):
        """Initialize a TimeRange; the class docstring describes the parameters."""
        if time_range and (start or stop):
            raise ValueError("Cannot specify both time_range and start/stop")

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
            return "Entire Observation Period"
        elif not self.start:
            return f"Before {self.stop}"
        elif not self.stop:
            return f"After {self.start}"
        else:
            return f"{self.start} to {self.stop}"

    def __iter__(self):
        return iter([self.start, self.stop])

    def __contains__(self, item):
        if not any([self.start, self.stop]):
            # Entire Period - True
            return True
        # stop is exclusive: TimeRange('2024') stops at 2025-01-01 00:00,
        # which belongs to the next period
        if not self.start:
            # Before stop - True if item < stop
            return item < self.stop
        if not self.stop:
            # After start - True if start <= item
            return self.start <= item
        return self.start <= item < self.stop

    @property
    def start(self) -> dt.datetime | None:
        """
        The start of the range, or None if it is open at the start.

        Setting it takes a datetime, a ``numpy.datetime64``, None, or an ISO
        8601 string, which stands for the start of the period it names.

        See Also
        --------
        TimeRange.stop : The end of the range.

        Examples
        --------
        >>> from lair.clock import TimeRange
        >>> TimeRange("2024-07").start
        datetime.datetime(2024, 7, 1, 0, 0)
        """
        return self._start

    @start.setter
    def start(self, start):
        if start is None or isinstance(start, dt.datetime):
            self._start = start
        elif isinstance(start, str):
            self._start = TimeRange.parse_iso(start)
        elif isinstance(start, np.datetime64):
            self._start = pd.to_datetime(start).to_pydatetime()
        else:
            raise ValueError("Invalid start time format")

    @property
    def stop(self) -> dt.datetime | None:
        """
        The end of the range, or None if it is open at the end.

        The stop itself is not in the range. Setting it takes a datetime, a
        ``numpy.datetime64``, None, or an ISO 8601 string, which stands for the
        end of the period it names.

        See Also
        --------
        TimeRange.start : The start of the range.

        Examples
        --------
        >>> from lair.clock import TimeRange
        >>> TimeRange("2024-07").stop
        datetime.datetime(2024, 8, 1, 0, 0)
        """
        return self._stop

    @stop.setter
    def stop(self, stop):
        if stop is None or isinstance(stop, dt.datetime):
            self._stop = stop
        elif isinstance(stop, str):
            self._stop = TimeRange.parse_iso(stop, inclusive=True)
        elif isinstance(stop, np.datetime64):
            self._stop = pd.to_datetime(stop).to_pydatetime()
        else:
            raise ValueError("Invalid stop time format")

    @property
    def total_seconds(self) -> float:
        """
        The length of the range in seconds; both ends must be set.

        Raises
        ------
        ValueError
            If the range is open at either end.

        See Also
        --------
        dt2decimalDate : Uses the length of a year from it.

        Examples
        --------
        >>> from lair.clock import TimeRange
        >>> TimeRange("2024-07").total_seconds / 86400  # days in July
        31.0
        """
        if self.start is None or self.stop is None:
            raise ValueError("Both start and stop times must be specified")
        return (self.stop - self.start).total_seconds()

    @staticmethod
    def parse_iso(string: str, inclusive: bool = False) -> dt.datetime:
        """
        Parse an ISO 8601 time string, such as ``"2024-07"``, into a datetime.

        Parameters
        ----------
        string : str
            The ISO 8601 time string, from a year down to fractions of a second.
            A trailing ``Z`` is accepted; the result has no time zone.
        inclusive : bool, optional
            If False (the default), return the start of the period the string
            names. If True, return its end, which is the start of the next
            period: ``"2024-07"`` gives 2024-08-01 00:00.

        Returns
        -------
        dt.datetime
            The parsed datetime object.

        Raises
        ------
        ValueError
            If the time_str format is invalid.

        See Also
        --------
        TimeRange : A range made from such strings.

        Examples
        --------
        >>> from lair.clock import TimeRange
        >>> TimeRange.parse_iso("2024-07")
        datetime.datetime(2024, 7, 1, 0, 0)
        >>> TimeRange.parse_iso("2024-07", inclusive=True)
        datetime.datetime(2024, 8, 1, 0, 0)
        >>> TimeRange.parse_iso("2024-07-04T12Z")
        datetime.datetime(2024, 7, 4, 12, 0)
        """
        # Parse time_range string using regex assuming ISO8601 format.
        # Each component is only allowed after the one before it, and the
        # whole string must match, so e.g. '2024-1' is rejected instead of
        # being read as hour 1 of 2024.
        iso8601 = (
            r"(?P<year>\d{4})"
            r"(?:-?(?P<month>\d{2})"
            r"(?:-?(?P<day>\d{2})"
            r"(?:[T\s]?(?P<hour>\d{2})"
            r"(?::?(?P<minute>\d{2})"
            r"(?::?(?P<second>\d{2}(?:\.\d{1,6})?))?)?)?)?)?"
            r"Z?"  # UTC designator; times are naive (UTC by convention)
        )
        match = re.fullmatch(iso8601, string.strip())
        if not match:
            raise ValueError(f"Invalid time string format: {string!r}")

        components = match.groupdict()
        year = int(components["year"])
        month = int(components["month"] or 1)
        day = int(components["day"] or 1)
        hour = int(components["hour"] or 0)
        minute = int(components["minute"] or 0)
        second, _, fraction = (components["second"] or "0").partition(".")
        microsecond = int(fraction.ljust(6, "0")) if fraction else 0

        start = dt.datetime(year, month, day, hour, minute, int(second), microsecond)

        # Determine the stop time based on the inclusive flag
        if inclusive:
            if components["year"] and not components["month"]:
                "YYYY"
                stop = dt.datetime(year + 1, 1, 1)
            elif components["month"] and not components["day"]:
                "YYYY-MM"
                mm = month + 1 if month < 12 else 1
                yyyy = year + 1 if month == 12 else year
                stop = dt.datetime(yyyy, mm, 1)
            elif components["day"] and not components["hour"]:
                "YYYY-MM-DD"
                stop = start + dt.timedelta(days=1)
            elif components["hour"] and not components["minute"]:
                "YYYY-MM-DDTHH"
                stop = start + dt.timedelta(hours=1)
            elif components["minute"] and not components["second"]:
                "YYYY-MM-DDTHH:MM"
                stop = start + dt.timedelta(minutes=1)
            elif components["second"]:
                "YYYY-MM-DDTHH:MM:SS"
                # one unit of the finest digit given, e.g. 12:30:45.5 -> +0.1 s
                stop = start + dt.timedelta(seconds=10.0 ** -len(fraction))
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

    Parameters
    ----------
    name : str, optional
        A name to add the elapsed times up under, in the class's ``timers``
        dict (shared by every Timer). By default None: not recorded.
    text : str, optional
        The message reported when the timer stops, formatted with the elapsed
        seconds. By default ``"Elapsed time: {:0.4f} seconds"``.
    logger : Callable[[str], None] | None, optional
        The function that reports the message, by default ``print``. None
        reports nothing.

    See Also
    --------
    Timer.start : Start it by hand.
    Timer.stop : Stop it by hand.

    Examples
    --------
    As a context manager:

    >>> from lair.clock import Timer
    >>> with Timer(text="took {:.0f} s"):
    ...     total = sum(range(1000))
    took 0 s

    By hand, without a report:

    >>> timer = Timer(logger=None)
    >>> timer.start()
    >>> elapsed = timer.stop()
    >>> elapsed >= 0
    True
    """

    timers: ClassVar[dict[str, float]] = {}
    name: str | None = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: Callable[[str], None] | None = print
    _start_time: float | None = field(default=None, init=False, repr=False)

    class TimerError(Exception):
        """A custom exception used to report errors in use of Timer class"""

        pass

    def __post_init__(self) -> None:
        """Add the timer to the dict of timers."""
        if self.name:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """
        Start a new timer

        Raises
        ------
        Timer.TimerError
            If the timer is already running.

        See Also
        --------
        Timer.stop : Stop it and report the time.
        """
        if self._start_time is not None:
            raise Timer.TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """
        Stop the timer, and report the elapsed time

        Returns
        -------
        float
            The elapsed time, in seconds. It is also added to ``timers`` under
            the timer's ``name``, if it has one.

        Raises
        ------
        Timer.TimerError
            If the timer is not running.

        See Also
        --------
        Timer.start : Start it.
        """
        if self._start_time is None:
            raise Timer.TimerError("Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] = self.timers.get(self.name, 0) + elapsed_time

        return elapsed_time

    def reset_timers(self):
        """
        Reset class timers

        Empties ``timers``, the elapsed times of every named Timer.

        See Also
        --------
        Timer.stop : Adds to them.
        """
        # Clear in place: existing Timer instances share this dict
        Timer.timers.clear()

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.stop()


def datetime_accessor(obj, accessor="dt"):
    """
    Return the datetime accessor of the object.

    A Series of times has its datetime methods under ``.dt``; a DatetimeIndex
    has them itself. This returns whichever ``obj`` has, so code can call
    ``tz_localize`` and the rest on either.

    Parameters
    ----------
    obj : pd.Series | pd.DatetimeIndex | Any
        The times.
    accessor : str, optional
        The name of the accessor, by default ``'dt'``.

    Returns
    -------
    Any
        ``obj.dt`` if ``obj`` has it, else ``obj``.

    See Also
    --------
    convert_timezones : Uses it for a Series or an index.

    Examples
    --------
    >>> import pandas as pd
    >>> from lair.clock import datetime_accessor
    >>> times = pd.Series(pd.date_range("2024-01-01", periods=2, freq="h"))
    >>> type(datetime_accessor(times)).__name__
    'DatetimeProperties'
    >>> type(datetime_accessor(pd.DatetimeIndex(times))).__name__
    'DatetimeIndex'
    """
    if hasattr(obj, accessor):
        return getattr(obj, accessor)
    return obj


def regular_times_to_intervals(
    times,
    time_step="monthly",
    closed: Literal["left", "right", "both", "neither"] = "left",
) -> pd.IntervalIndex:
    """
    Convert an array of regular times to intervals of the specified length.

    Parameters
    ----------
    times : np.array
        A numpy array of datetime64 values indicating the start of each interval.
    time_step : str, optional
        The interval step ('hourly', 'daily', 'monthly', 'annual').
        Determines the length of each interval. Default is 'monthly'.
    closed : str, optional
        Defines if the interval is closed on the 'left', 'right', 'both', or 'neither'. Default is 'left'.

    Returns
    -------
    pd.IntervalIndex
        An index with intervals covering the specified time range.

    Raises
    ------
    ValueError
        If `time_step` is not one of 'hourly', 'daily', 'monthly', or 'annual'.

    See Also
    --------
    TimeRange : One interval, from ISO strings.

    Examples
    --------
    >>> import pandas as pd
    >>> from lair.clock import regular_times_to_intervals
    >>> months = pd.date_range("2024-01-01", periods=2, freq="MS")
    >>> intervals = regular_times_to_intervals(months)
    >>> [str(interval) for interval in intervals]
    ['[2024-01-01 00:00:00, 2024-02-01 00:00:00)', '[2024-02-01 00:00:00, 2024-03-01 00:00:00)']
    """
    # Convert the numpy array to pandas DatetimeIndex
    start_times = pd.to_datetime(times)

    # Define a dictionary mapping time_step to DateOffset
    offsets = {
        "hourly": pd.offsets.DateOffset(hours=1),
        "daily": pd.offsets.DateOffset(days=1),
        "monthly": pd.offsets.DateOffset(months=1),
        "annual": pd.offsets.DateOffset(years=1),
    }

    # Get the offset from the dictionary or raise an error if the time_step is invalid
    if time_step not in offsets:
        raise ValueError("time_step must be 'hourly', 'daily', 'monthly', or 'annual'")
    offset = offsets[time_step]

    # Calculate the end times for each interval
    stop_times = start_times + offset

    # Create the IntervalIndex with specified closure
    intervals = pd.IntervalIndex.from_arrays(start_times, stop_times, closed=closed)
    return intervals


# ----- Time Aggregation ----- #


def diurnal(
    data: pd.DataFrame,
    freq: str = "1h",
    statistic: str | list[str] = "mean",
    method: Literal["floor", "ceil", "round"] = "floor",
):
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
        The aggregated data, indexed by the time of day.

    See Also
    --------
    seasonal : Aggregates by season and year.

    Examples
    --------
    The mean of two days of hourly data at each hour of the day:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from lair.clock import diurnal
    >>> times = pd.date_range("2024-01-01", periods=48, freq="h")
    >>> data = pd.DataFrame({"ch4": np.arange(48.0)}, index=times)
    >>> diurnal(data).head(2)
               ch4
    00:00:00  12.0
    01:00:00  13.0
    """
    resolution = getattr(data.index, method)(freq).time
    agg = data.groupby(resolution).agg(statistic)

    return agg


def seasonal(data: pd.DataFrame, statistic: str | list[str] = "mean") -> pd.DataFrame:
    """
    Aggregate data by season and year.

    Seasons are DJF, MAM, JJA and SON. DJF is labeled by the year of its
    January and February: DJF 2024 is Dec 2023 + Jan 2024 + Feb 2024 (December
    rows count towards the next year). The other seasons keep their calendar
    year.

    Partial seasons at the edges of the data are kept, not dropped. For data
    covering calendar year 2024, DJF 2024 is Jan-Feb 2024 only and DJF 2025 is
    Dec 2024 only.

    Parameters
    ----------
    data : pd.DataFrame
        The data to aggregate. Panda DataFrame with a DateTimeIndex.
    statistic : str, optional
        The statistic to compute for each group. Default is 'mean'.

    Returns
    -------
    pd.DataFrame
        The aggregated data, indexed by (``season``, ``year``).

    See Also
    --------
    SEASONS : The season of each month.
    diurnal : Aggregates by the time of day.

    Examples
    --------
    Monthly data for 2024, where each month's value is its number minus one:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from lair.clock import seasonal
    >>> months = pd.date_range("2024-01-01", periods=12, freq="MS")
    >>> seasons = seasonal(pd.DataFrame({"value": np.arange(12.0)}, index=months))
    >>> [f"{season} {year}" for season, year in seasons.index]
    ['DJF 2024', 'MAM 2024', 'JJA 2024', 'SON 2024', 'DJF 2025']
    >>> seasons["value"].tolist()
    [0.5, 3.0, 6.0, 9.0, 11.0]
    """
    # Quarters starting in Dec/Mar/Jun/Sep line up with DJF/MAM/JJA/SON
    df = data.resample("QS-DEC").agg(statistic)  # pyrefly: ignore[no-matching-overload]
    index = pd.DatetimeIndex(df.index)
    season = index.month.map(SEASONS)

    # A DJF bin starts on Dec 1, so its label is the following (Jan/Feb) year
    year = index.year + (index.month == 12)
    return df.set_axis(
        pd.MultiIndex.from_arrays([season, year], names=["season", "year"])
    )


# ----- Decimal Time ----- #


def dt2decimalDate(datetime: dt.datetime) -> float:
    """
    Convert a datetime object to a decimal date.

    Naive datetimes are used as-is (by convention they are UTC). Timezone-aware
    datetimes (incl. tz-aware ``pd.Timestamp``) are converted to UTC first, so
    the decimal date counts elapsed time from Jan 1 00:00 UTC of the UTC year.
    This gives the same answer for pytz and zoneinfo zones, and keeps the two
    instants of a DST fall-back hour distinct and in order.

    Parameters
    ----------
    datetime : dt.datetime
        The datetime object to convert.

    Returns
    -------
    float
        The decimal date.

    See Also
    --------
    decimalDate2dt : The other way.

    Examples
    --------
    >>> import datetime as dt
    >>> from lair.clock import dt2decimalDate
    >>> round(dt2decimalDate(dt.datetime(2024, 7, 1)), 4)  # 182 of 366 days
    2024.4973
    """
    if datetime.utcoffset() is not None:
        datetime = datetime.astimezone(dt.UTC).replace(tzinfo=None)

    this_year = dt.datetime(datetime.year, 1, 1)
    total_seconds = (datetime - this_year).total_seconds()
    total_seconds_year = TimeRange(str(datetime.year)).total_seconds
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
        The naive datetime object. For a decimal date made from a tz-aware
        datetime by :func:`dt2decimalDate`, this is the UTC time.

    See Also
    --------
    dt2decimalDate : The other way.

    Examples
    --------
    >>> from lair.clock import decimalDate2dt
    >>> decimalDate2dt(2024.5)  # half of 366 days
    datetime.datetime(2024, 7, 2, 0, 0)
    """
    year = int(decimalDate)
    rem = decimalDate - year

    this_year = dt.datetime(year, 1, 1)
    total_seconds_year = TimeRange(str(year)).total_seconds
    # Round to the millisecond: the float year term only carries ~1e-5 s of
    # precision, and without rounding midnight comes back as 23:59:59.999998
    # of the day before
    return this_year + dt.timedelta(seconds=round(total_seconds_year * rem, 3))


# ----- Time Zones ----- #

# The ``ambiguous`` and ``nonexistent`` options of ``convert_timezones`` and its
# shortcuts: pandas' ``tz_localize`` options.
_Ambiguous = (
    Literal["raise", "NaT", "infer"] | bool | Sequence[bool] | npt.NDArray[np.bool_]
)
_Nonexistent = Literal["raise", "NaT", "shift_forward", "shift_backward"] | dt.timedelta


def convert_timezones(
    x,
    totz,
    fromtz=None,
    localize=False,
    driver=None,
    ambiguous: _Ambiguous = "raise",
    nonexistent: _Nonexistent = "raise",
):
    """
    Convert the times from one timezone to another.

    Parameters
    ----------
    x : dt.datetime | pd.Timestamp | list[dt.datetime] | pd.DataFrame | pd.Series
        The times to convert. With ``driver='pandas'``, the index of a
        DataFrame or the values of a Series. Without it, a single time or a
        list (or other iterable) of them.
    totz : str
        The timezone to convert the times to.
    fromtz : str, optional
        The timezone of the naive input times. Required if any input time is
        naive (tz-aware times are converted from their own timezone).
    localize : bool, optional
        If True, the times will be localized to the totz timezone, by default False
    driver : None | 'pandas', optional
        None for a list of datetimes, 'pandas' for a DataFrame or Series.
    ambiguous : 'raise' | 'NaT' | 'infer' | bool | array of bool, optional
        How to localize naive times that happen twice in ``fromtz`` (DST
        fall-back), with pandas ``tz_localize`` semantics: True is the DST
        (first) instant, False the standard-time (second) one, 'NaT' gives NaT,
        and 'raise' (the default) raises a ValueError. An array gives one flag
        per input time (for a list, aware elements' flags are ignored).
    nonexistent : 'raise' | 'NaT' | 'shift_forward' | 'shift_backward' | timedelta, optional
        How to localize naive times that don't exist in ``fromtz`` (DST
        spring-forward), with pandas ``tz_localize`` semantics. Default 'raise'.

    Returns
    -------
    dt.datetime | pd.Timestamp | list[dt.datetime] | pd.DataFrame | pd.Series
        The converted times, in the shape of ``x``. A single time gives a single
        time, converted the way each element of a list is: a ``datetime`` stays
        a ``datetime`` and a ``pd.Timestamp`` stays a ``pd.Timestamp`` (``pd.NaT``
        if ``ambiguous`` or ``nonexistent`` is ``'NaT'`` and the time is one of
        those).

    Raises
    ------
    ValueError
        If there are naive times and no ``fromtz``, or a naive time is ambiguous
        or nonexistent in ``fromtz`` and ``ambiguous``/``nonexistent`` is 'raise',
        or ``driver='pandas'`` and ``x`` is not a DataFrame or Series.

    See Also
    --------
    UTC2MTN : This conversion from UTC to Salt Lake City time, ready-made.
    diurnal : Average by hour of the day, once the data is in local time.
    AFTERNOON : The hours to select once the data is in local time.

    Examples
    --------
    18:00 UTC in January is 11:00 in Salt Lake City:

    >>> import datetime as dt
    >>> from lair.clock import convert_timezones
    >>> convert_timezones(dt.datetime(2024, 1, 15, 18), "America/Denver", fromtz="UTC")
    datetime.datetime(2024, 1, 15, 11, 0, tzinfo=zoneinfo.ZoneInfo(key='America/Denver'))

    A list gives a list, converted time by time:

    >>> convert_timezones(
    ...     [dt.datetime(2024, 1, 15, 18)], "America/Denver", fromtz="UTC"
    ... )
    [datetime.datetime(2024, 1, 15, 11, 0, tzinfo=zoneinfo.ZoneInfo(key='America/Denver'))]

    The index of a DataFrame, kept naive in local time with ``localize=True``:

    >>> import pandas as pd
    >>> data = pd.DataFrame(
    ...     {"ch4": [2000.0]}, index=pd.to_datetime(["2024-07-15 18:00"])
    ... )
    >>> convert_timezones(
    ...     data, "America/Denver", fromtz="UTC", localize=True, driver="pandas"
    ... )
                            ch4
    2024-07-15 12:00:00  2000.0
    """
    if driver is None:
        # A single time: convert it as a list of one, and give back the one
        if isinstance(x, dt.datetime):
            return convert_timezones(
                [x], totz, fromtz, localize, None, ambiguous, nonexistent
            )[0]

        times = list(x)

        # Assign the fromtz timezone to the naive times (aware times keep theirs).
        # Localize with pandas so DST transitions behave like the pandas driver.
        naive = [i for i, t in enumerate(times) if t.tzinfo is None]
        if naive:
            if fromtz is None:
                raise ValueError("Naive times need fromtz: the timezone they are in")
            if np.ndim(ambiguous) > 0:
                # one flag per input time -> keep the naive ones
                ambiguous = np.asarray(ambiguous)[naive]
            localized = pd.DatetimeIndex([times[i] for i in naive]).tz_localize(
                fromtz,
                # pandas-stubs leave out the scalar bool and the list it accepts
                # pyrefly: ignore[bad-argument-type]
                ambiguous=ambiguous,
                nonexistent=nonexistent,
            )
            for i, t in zip(naive, localized, strict=True):
                if t is pd.NaT or isinstance(times[i], pd.Timestamp):
                    times[i] = t
                else:
                    times[i] = t.to_pydatetime()

        # Convert the times to the specified timezone.
        totz = ZoneInfo(totz)
        converted_times = [t if t is pd.NaT else t.astimezone(totz) for t in times]

        # If localize is True, localize the times to the totz timezone.
        if localize:
            converted_times = [
                t if t is pd.NaT else t.replace(tzinfo=None) for t in converted_times
            ]

        return converted_times
    elif driver == "pandas":
        # Check the type before copying: a list's copy() takes no 'deep'
        if not isinstance(x, (pd.DataFrame, pd.Series)):
            raise ValueError("x is not a DataFrame or Series")
        data = x.copy(deep=True)

        # If x is a DataFrame, convert DateTimeIndex
        # If x is a Series, convert the series
        if isinstance(data, pd.DataFrame):
            times = data.index
            index = True
        else:
            times = data
            index = False

        # If the times are not tz-aware, assign them the fromtz timezone.
        if datetime_accessor(times).tz is None:
            if fromtz is None:
                raise ValueError("Naive times need fromtz: the timezone they are in")
            times = datetime_accessor(times).tz_localize(
                fromtz, ambiguous=ambiguous, nonexistent=nonexistent
            )

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


# The shortcuts are functions, not ``functools.partial`` objects, so that each has
# its own signature and its own ``help()`` text.

# Mountain Standard Time, a fixed UTC-7 (POSIX signs are reversed). Not 'MST': since
# tzdata 2024b that is a link to America/Phoenix, which kept daylight saving time in
# 1918-1919, 1942-1944 and 1967.
_MST = "Etc/GMT+7"


def UTC2(
    x,
    totz,
    *,
    localize=False,
    driver=None,
    ambiguous: _Ambiguous = "raise",
    nonexistent: _Nonexistent = "raise",
):
    """
    Convert times from UTC to the time zone ``totz``.

    A time without a time zone is taken as UTC (the rule on the
    :ref:`clock example <utc-rule>`), and a time with one is converted from its
    own. :func:`UTC2MST` and :func:`UTC2MTN` are the two common cases.

    Parameters
    ----------
    x : dt.datetime | pd.Timestamp | list[dt.datetime] | pd.DataFrame | pd.Series
        The times: a single time, a list of them, or with ``driver='pandas'``
        the index of a DataFrame or the values of a Series.
    totz : str
        The time zone to convert to, such as ``'America/Denver'``.
    localize : bool, optional
        If True, drop the time zone from the result and keep the clock time of
        ``totz``. Default False.
    driver : None | 'pandas', optional
        ``'pandas'`` for a DataFrame or Series; None for a single time or a list.
    ambiguous, nonexistent : optional
        As in :func:`convert_timezones`. UTC has no repeated or skipped hours, so
        they never apply; every shortcut takes them.

    Returns
    -------
    dt.datetime | pd.Timestamp | list[dt.datetime] | pd.DataFrame | pd.Series
        The converted times, in the shape of ``x``: a single time gives a single
        time, a ``datetime`` a ``datetime`` and a ``pd.Timestamp`` a
        ``pd.Timestamp``.

    See Also
    --------
    convert_timezones : The general conversion, from any zone to any zone.
    UTC2MTN : UTC to Salt Lake City time.
    UTC2MST : UTC to Mountain Standard Time.

    Examples
    --------
    18:00 UTC in Pacific time, in July:

    >>> import datetime as dt
    >>> from lair.clock import UTC2
    >>> UTC2(dt.datetime(2024, 7, 15, 18), "America/Los_Angeles")
    datetime.datetime(2024, 7, 15, 11, 0, tzinfo=zoneinfo.ZoneInfo(key='America/Los_Angeles'))
    """
    return convert_timezones(
        x,
        totz,
        fromtz="UTC",
        localize=localize,
        driver=driver,
        ambiguous=ambiguous,
        nonexistent=nonexistent,
    )


def UTC2MST(
    x,
    *,
    localize=False,
    driver=None,
    ambiguous: _Ambiguous = "raise",
    nonexistent: _Nonexistent = "raise",
):
    """
    Convert times from UTC to Mountain Standard Time (``'Etc/GMT+7'``).

    MST is 7 hours behind UTC all year, with no daylight saving time, so every
    day is 24 hours long, in every year. The zone is ``'Etc/GMT+7'``, a fixed
    offset (its sign is POSIX's, reversed): the tz database's ``'MST'`` is
    Phoenix's time, which kept daylight saving time in 1967 and the war years. A time without a time zone is taken as UTC (the rule on
    the :ref:`clock example <utc-rule>`), and a time with one is converted from
    its own.

    Parameters
    ----------
    x : dt.datetime | pd.Timestamp | list[dt.datetime] | pd.DataFrame | pd.Series
        The times: a single time, a list of them, or with ``driver='pandas'``
        the index of a DataFrame or the values of a Series.
    localize : bool, optional
        If True, drop the time zone from the result and keep the clock time in
        MST. Default False.
    driver : None | 'pandas', optional
        ``'pandas'`` for a DataFrame or Series; None for a single time or a list.
    ambiguous, nonexistent : optional
        As in :func:`convert_timezones`. UTC has no repeated or skipped hours, so
        they never apply; every shortcut takes them.

    Returns
    -------
    dt.datetime | pd.Timestamp | list[dt.datetime] | pd.DataFrame | pd.Series
        The converted times, in the shape of ``x``: a single time gives a single
        time, a ``datetime`` a ``datetime`` and a ``pd.Timestamp`` a
        ``pd.Timestamp``.

    See Also
    --------
    UTC2MTN : UTC to Salt Lake City time, which follows daylight saving time.
    MST2UTC : The way back, from MST to UTC.
    convert_timezones : The general conversion, from any zone to any zone.

    Examples
    --------
    18:00 UTC is 11:00 MST, in January and in July:

    >>> import datetime as dt
    >>> from lair.clock import UTC2MST
    >>> UTC2MST(dt.datetime(2024, 7, 15, 18))
    datetime.datetime(2024, 7, 15, 11, 0, tzinfo=zoneinfo.ZoneInfo(key='Etc/GMT+7'))

    The index of a DataFrame, kept naive in MST with ``localize=True``:

    >>> import pandas as pd
    >>> data = pd.DataFrame(
    ...     {"ch4": [2000.0]}, index=pd.to_datetime(["2024-07-15 18:00"])
    ... )
    >>> UTC2MST(data, localize=True, driver="pandas")
                            ch4
    2024-07-15 11:00:00  2000.0
    """
    return UTC2(
        x,
        _MST,
        localize=localize,
        driver=driver,
        ambiguous=ambiguous,
        nonexistent=nonexistent,
    )


def UTC2MTN(
    x,
    *,
    localize=False,
    driver=None,
    ambiguous: _Ambiguous = "raise",
    nonexistent: _Nonexistent = "raise",
):
    """
    Convert times from UTC to Salt Lake City time (``'America/Denver'``).

    The result follows daylight saving time: 6 hours behind UTC in summer, 7 in
    winter. A time without a time zone is taken as UTC (the rule on the
    :ref:`clock example <utc-rule>`), and a time with one is converted from its
    own.

    Parameters
    ----------
    x : dt.datetime | pd.Timestamp | list[dt.datetime] | pd.DataFrame | pd.Series
        The times: a single time, a list of them, or with ``driver='pandas'``
        the index of a DataFrame or the values of a Series.
    localize : bool, optional
        If True, drop the time zone from the result and keep the clock time in
        Salt Lake City. Default False.
    driver : None | 'pandas', optional
        ``'pandas'`` for a DataFrame or Series; None for a single time or a list.
    ambiguous, nonexistent : optional
        As in :func:`convert_timezones`. UTC has no repeated or skipped hours, so
        they never apply; every shortcut takes them.

    Returns
    -------
    dt.datetime | pd.Timestamp | list[dt.datetime] | pd.DataFrame | pd.Series
        The converted times, in the shape of ``x``: a single time gives a single
        time, a ``datetime`` a ``datetime`` and a ``pd.Timestamp`` a
        ``pd.Timestamp``.

    See Also
    --------
    UTC2MST : UTC to Mountain Standard Time, with no daylight saving time.
    MTN2UTC : The way back, from Salt Lake City time to UTC.
    convert_timezones : The general conversion, from any zone to any zone.

    Examples
    --------
    18:00 UTC in July is 12:00 daylight time:

    >>> import datetime as dt
    >>> from lair.clock import UTC2MTN
    >>> UTC2MTN(dt.datetime(2024, 7, 15, 18))
    datetime.datetime(2024, 7, 15, 12, 0, tzinfo=zoneinfo.ZoneInfo(key='America/Denver'))

    A ``pd.Timestamp`` gives a ``pd.Timestamp``:

    >>> import pandas as pd
    >>> UTC2MTN(pd.Timestamp("2024-07-15 18:00"))
    Timestamp('2024-07-15 12:00:00-0600', tz='America/Denver')

    The index of a DataFrame, kept naive in local time with ``localize=True``:

    >>> data = pd.DataFrame(
    ...     {"ch4": [2000.0]}, index=pd.to_datetime(["2024-07-15 18:00"])
    ... )
    >>> UTC2MTN(data, localize=True, driver="pandas")
                            ch4
    2024-07-15 12:00:00  2000.0
    """
    return UTC2(
        x,
        "America/Denver",
        localize=localize,
        driver=driver,
        ambiguous=ambiguous,
        nonexistent=nonexistent,
    )


def MST2UTC(
    x,
    *,
    localize=False,
    driver=None,
    ambiguous: _Ambiguous = "raise",
    nonexistent: _Nonexistent = "raise",
):
    """
    Convert times from Mountain Standard Time (``'Etc/GMT+7'``) to UTC.

    Use it for a file whose times are standard time all year (its manual or
    settings say so): MST has no daylight saving time, so no time is repeated or
    skipped. A time without a time zone is taken as MST, UTC-7 in every year (see
    :func:`UTC2MST`), and a time with one is converted from its own.

    Parameters
    ----------
    x : dt.datetime | pd.Timestamp | list[dt.datetime] | pd.DataFrame | pd.Series
        The times: a single time, a list of them, or with ``driver='pandas'``
        the index of a DataFrame or the values of a Series.
    localize : bool, optional
        If True, drop the time zone from the result and keep the UTC clock time.
        Default False.
    driver : None | 'pandas', optional
        ``'pandas'`` for a DataFrame or Series; None for a single time or a list.
    ambiguous, nonexistent : optional
        As in :func:`convert_timezones`. MST has no repeated or skipped hours, so
        they never apply; every shortcut takes them.

    Returns
    -------
    dt.datetime | pd.Timestamp | list[dt.datetime] | pd.DataFrame | pd.Series
        The converted times, in the shape of ``x``: a single time gives a single
        time, a ``datetime`` a ``datetime`` and a ``pd.Timestamp`` a
        ``pd.Timestamp``.

    See Also
    --------
    MTN2UTC : The same from Salt Lake City time, which follows daylight saving time.
    UTC2MST : The way back, from UTC to MST.
    convert_timezones : The general conversion, from any zone to any zone.

    Examples
    --------
    11:00 MST is 18:00 UTC:

    >>> import datetime as dt
    >>> from lair.clock import MST2UTC
    >>> MST2UTC(dt.datetime(2024, 7, 15, 11))
    datetime.datetime(2024, 7, 15, 18, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC'))

    The index of a DataFrame, kept naive in UTC with ``localize=True``:

    >>> import pandas as pd
    >>> data = pd.DataFrame(
    ...     {"ch4": [2000.0]}, index=pd.to_datetime(["2024-07-15 11:00"])
    ... )
    >>> MST2UTC(data, localize=True, driver="pandas")
                            ch4
    2024-07-15 18:00:00  2000.0
    """
    return convert_timezones(
        x,
        "UTC",
        fromtz=_MST,
        localize=localize,
        driver=driver,
        ambiguous=ambiguous,
        nonexistent=nonexistent,
    )


def MTN2UTC(
    x,
    *,
    localize=False,
    driver=None,
    ambiguous: _Ambiguous = "raise",
    nonexistent: _Nonexistent = "raise",
):
    """
    Convert times from Salt Lake City time (``'America/Denver'``) to UTC.

    A time without a time zone is taken as Salt Lake City clock time, and a time
    with one is converted from its own. Daylight saving time repeats one hour in
    November and skips one in March, so a clock-time file can hold a time that
    cannot be converted without a choice: ``ambiguous`` and ``nonexistent`` make
    it (the :ref:`clock example <dst-errors>` shows both).

    Parameters
    ----------
    x : dt.datetime | pd.Timestamp | list[dt.datetime] | pd.DataFrame | pd.Series
        The times: a single time, a list of them, or with ``driver='pandas'``
        the index of a DataFrame or the values of a Series.
    localize : bool, optional
        If True, drop the time zone from the result and keep the UTC clock time.
        Default False.
    driver : None | 'pandas', optional
        ``'pandas'`` for a DataFrame or Series; None for a single time or a list.
    ambiguous : 'raise' | 'NaT' | 'infer' | bool | array of bool, optional
        What to do with a time that happens twice (the clocks go back): True is
        the first (daylight time), False the second (standard time). The default
        'raise' raises a ``ValueError``. As in :func:`convert_timezones`.
    nonexistent : 'raise' | 'NaT' | 'shift_forward' | 'shift_backward' | timedelta, optional
        What to do with a time that never happens (the clocks go forward). The
        default 'raise' raises a ``ValueError``. As in :func:`convert_timezones`.

    Returns
    -------
    dt.datetime | pd.Timestamp | list[dt.datetime] | pd.DataFrame | pd.Series
        The converted times, in the shape of ``x``: a single time gives a single
        time, a ``datetime`` a ``datetime`` and a ``pd.Timestamp`` a
        ``pd.Timestamp``.

    Raises
    ------
    ValueError
        If a time is ambiguous or nonexistent and the option for it is 'raise'.

    See Also
    --------
    MST2UTC : The same from standard time, where nothing is repeated or skipped.
    UTC2MTN : The way back, from UTC to Salt Lake City time.
    convert_timezones : The general conversion, from any zone to any zone.

    Examples
    --------
    12:00 in Salt Lake City in July is 18:00 UTC:

    >>> import datetime as dt
    >>> from lair.clock import MTN2UTC
    >>> MTN2UTC(dt.datetime(2024, 7, 15, 12))
    datetime.datetime(2024, 7, 15, 18, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC'))

    The clocks went back on 2024-11-03, so 01:30 happened twice; say which:

    >>> MTN2UTC(dt.datetime(2024, 11, 3, 1, 30), ambiguous=True)
    datetime.datetime(2024, 11, 3, 7, 30, tzinfo=zoneinfo.ZoneInfo(key='UTC'))
    >>> MTN2UTC(dt.datetime(2024, 11, 3, 1, 30), ambiguous=False)
    datetime.datetime(2024, 11, 3, 8, 30, tzinfo=zoneinfo.ZoneInfo(key='UTC'))

    The index of a DataFrame, kept naive in UTC with ``localize=True``:

    >>> import pandas as pd
    >>> data = pd.DataFrame(
    ...     {"ch4": [2000.0]}, index=pd.to_datetime(["2024-07-15 12:00"])
    ... )
    >>> MTN2UTC(data, localize=True, driver="pandas")
                            ch4
    2024-07-15 18:00:00  2000.0
    """
    return convert_timezones(
        x,
        "UTC",
        fromtz="America/Denver",
        localize=localize,
        driver=driver,
        ambiguous=ambiguous,
        nonexistent=nonexistent,
    )
