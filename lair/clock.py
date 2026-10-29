"""
Times and time zones.

:func:`convert_timezones` moves times between zones (a time without a zone
needs ``fromtz``), and :func:`UTC2MTN` and its siblings are its common cases.
The remaining functions turn times into decimal years (:func:`dt2decimalDate`),
intervals, time-difference matrices, and diurnal and seasonal aggregates.
:doc:`/examples/clock` shows the time-zone rules and the afternoon mean.
"""

import datetime as dt
import time
from collections.abc import Callable, Sequence
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = [
    "AFTERNOON",
    "SEASONS",
    "Timer",
    "datetime_accessor",
    "regular_times_to_intervals",
    "time_difference_matrix",
    "time_decay_matrix",
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
    pandas.interval_range : Intervals of one length between a start and an end.

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


def time_difference_matrix(times, absolute: bool = True) -> np.ndarray:
    """
    Calculate the time differences between each pair of times.

    Parameters
    ----------
    times : list[dt.datetime]
        The list of times to calculate the differences between.
    absolute : bool, optional
        If True, return the absolute differences. Default is True.

    Returns
    -------
    np.ndarray
        The matrix of time differences, as ``timedelta64``: row i, column j is
        time i minus time j (its absolute value by default).

    See Also
    --------
    time_decay_matrix : An exponential decay of these differences.

    Examples
    --------
    >>> import pandas as pd
    >>> from lair.clock import time_difference_matrix
    >>> times = pd.to_datetime(
    ...     ["2024-01-01 00:00", "2024-01-01 06:00", "2024-01-02 00:00"]
    ... )
    >>> time_difference_matrix(times) / pd.Timedelta("1h")
    array([[ 0.,  6., 24.],
           [ 6.,  0., 18.],
           [24., 18.,  0.]])
    """
    times = pd.DatetimeIndex(
        times
    )  # wrap in pandas DatetimeIndex as np.subtract.outer doesn't like pd.Series
    diffs = np.subtract.outer(times, times)
    if absolute:
        diffs = np.abs(diffs)
    return diffs


def time_decay_matrix(times, decay: str | pd.Timedelta) -> np.ndarray:
    """
    Calculate the time decay matrix for the specified times and decay.

    Parameters
    ----------
    times : list[dt.datetime]
        The list of times to calculate the decay matrix for.
    decay : str | pd.Timedelta
        The decay to use for the exponential decay: the time over which a
        value falls to 1/e.

    Returns
    -------
    np.ndarray
        The matrix of time decay values: ``exp(-|t_i - t_j| / decay)``, 1 on
        the diagonal.

    See Also
    --------
    time_difference_matrix : The differences it decays.

    Examples
    --------
    >>> import pandas as pd
    >>> from lair.clock import time_decay_matrix
    >>> times = pd.to_datetime(
    ...     ["2024-01-01 00:00", "2024-01-01 06:00", "2024-01-02 00:00"]
    ... )
    >>> time_decay_matrix(times, "1D").round(3)
    array([[1.   , 0.779, 0.368],
           [0.779, 1.   , 0.472],
           [0.368, 0.472, 1.   ]])
    """
    # Time differences (timedelta64) over the decay time scale -> exponential decay
    diffs = time_difference_matrix(times, absolute=True)
    return np.exp(-(diffs / pd.Timedelta(decay).to_timedelta64()))


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


def _seconds_in_year(year: int) -> float:
    """Return the number of seconds in the calendar year (366 days in a leap year)."""
    return (dt.datetime(year + 1, 1, 1) - dt.datetime(year, 1, 1)).total_seconds()


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
    total_seconds_year = _seconds_in_year(datetime.year)
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
    total_seconds_year = _seconds_in_year(year)
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
