"""
Background concentrations of a trace gas from its own time series.

:func:`get_well_mixed` keeps the afternoon hours, when the boundary layer is
deep; :func:`rolling_baseline` and :func:`phase_shift_corrected_baseline` take a
low quantile of a moving window; :func:`thoning` fits NOAA's curve (Thoning et
al., 1989) through the data. CarbonTracker's background is in
:mod:`lair.noaa`. :doc:`/examples/background` shows the methods side by side
and when to use which. The module needs NOAA's CCG filter, which the first
``import lair`` downloads (see :ref:`first-import-download`), and ``scipy``.
"""

import datetime as dt
import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    from lair._ccg_filter import ccgFilter  # make available to user
except ModuleNotFoundError as e:
    if e.name != "lair._ccg_filter":
        raise
    raise ImportError(
        "lair.background needs NOAA's CCG filter, which isn't installed. Run "
        "lair.setup_ccg_filter() (needs outbound FTP to ftp.gml.noaa.gov)."
    ) from e
from lair.clock import AFTERNOON, dt2decimalDate

logger = logging.getLogger(__name__)


def get_well_mixed(
    data: pd.Series | pd.DataFrame,
    hours: list[int] = AFTERNOON,
    *,
    tz: str,
) -> pd.Series | pd.DataFrame:
    """
    Average the data over the well-mixed hours of each day.

    The hours are read in the time zone ``tz``, which is always required: an
    index in UTC (all UATAQ data) would otherwise give the hours in UTC, which in
    the Americas are the morning, under the night-time inversion. A naive index
    is UTC, as everywhere in lair.

    Parameters
    ----------
    data : pd.Series | pd.DataFrame
        Time series data to subset. Must have a datetime index.
    hours : list[int], optional
        Hours of the day to keep, in ``tz``. Default is
        ``lair.clock.AFTERNOON``, 12 to 16 local standard time.
    tz : str
        The time zone ``hours`` are in, such as ``'MST'``. Keyword only. The
        index is converted to it first (a naive index from UTC), and each day
        runs from midnight in it; an index already in ``tz`` is unchanged. Give
        a standard-time zone (``'MST'``, not ``'America/Denver'``) for hours in
        local standard time, and ``'UTC'`` for UTC hours.

    Returns
    -------
    pd.Series | pd.DataFrame
        Daily means over the well-mixed hours (non-numeric columns are dropped),
        each labeled at midnight in ``tz``.

    Notes
    -----
    An index that holds local times without a time zone is read as UTC, so
    ``tz='MST'`` would shift it by 7 hours: localize it first
    (``data.tz_localize('MST')``).

    Examples
    --------
    Hourly data stamped in UTC, averaged over the afternoon in Mountain
    Standard Time:

    >>> import pandas as pd
    >>> from lair.background import get_well_mixed
    >>> times = pd.date_range("2024-06-01", periods=48, freq="h", tz="UTC")
    >>> local_hour = pd.Series(times.tz_convert("MST").hour, index=times)
    >>> get_well_mixed(local_hour, tz="MST")
    2024-06-01 00:00:00-07:00    14.0
    2024-06-02 00:00:00-07:00    14.0
    Freq: D, dtype: float64
    """
    index = pd.DatetimeIndex(data.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    index = index.tz_convert(tz)
    return (
        data.set_axis(index)[index.hour.isin(hours)]
        .resample("1D")
        .mean(numeric_only=True)
    )


def rolling_baseline(
    data: pd.Series,
    window: Any = "24h",
    q: float = 0.01,
    min_periods: int = 1,
    center: bool = True,
) -> pd.Series:
    """
    Calculate the baseline concentration as a rolling quantile.

    The baseline is the rolling ``q`` quantile over ``window``, smoothed with a
    rolling mean over the same window.

    Parameters
    ----------
    data : pd.Series
        Time series of data to calculate the baseline from.
    window : Any, optional
        Window size for the rolling calculation. Must be passable to
        pd.Timedelta. Default is '24h'.
    q : float, optional
        Quantile to calculate the baseline from. Default is 0.01.
    min_periods : int, optional
        Minimum number of periods within each window required to have a value.
        Default is 1.
    center : bool, optional
        Center the window on the timestamp. Default is True.

    Returns
    -------
    pd.Series
        Baseline concentration
    """
    window = pd.Timedelta(window)
    baseline = (
        data.rolling(window=window, center=center, min_periods=min_periods)
        .quantile(q)
        .rolling(window=window, center=center, min_periods=min_periods)
        .mean()
    )

    return baseline


def phase_shift_corrected_baseline(
    data: pd.Series, n: int = 3600, q: float = 0.01
) -> pd.Series:
    """
    Derive a baseline concentration from a low quantile, minimizing phase shift.

    This method uses forward-looking and backward-looking
    windows to better represent the lowest observed concentrations during
    periods of rapid change.

    .. note::
        Original developed by Ben Fasoli for the Google Street View project.
        See supplementary material in: https://doi.org/10.1016/j.atmosenv.2023.119995

    Parameters
    ----------
    data : pd.Series
        Signal time series with a datetime index. The baseline is calculated
        separately for each calendar day of the index, in the index's own time
        zone (the function does no time zone conversion), so pass local time to
        get local days.
    n : int, optional
        Window size in seconds. Default is 3600.
    q : float, optional
        Quantile to extract from signal. Default is 0.01.

    Returns
    -------
    pd.Series
        Baseline concentration
    """
    if not n % 2:
        n += 1

    b = []
    for _, y in data.groupby(pd.DatetimeIndex(data.index).floor("D")):
        hz = y.asfreq("s")
        left = hz.rolling(n, min_periods=1).quantile(q)
        right = hz.iloc[::-1].rolling(n, min_periods=1).quantile(q).iloc[::-1]

        forward = right.rolling(n, min_periods=1).max()
        backward = left.iloc[::-1].rolling(n, min_periods=1).max().iloc[::-1]

        b.append(
            pd.concat([forward, backward], axis=1)
            .max(axis=1)
            .rolling(n, min_periods=1, center=True, win_type="blackman")
            .mean()
        )

    return pd.concat(b)


def thoning_filter(data: pd.Series, **kwargs) -> ccgFilter:
    """
    Create a Thoning filter object from a time series of data.

    Parameters
    ----------
    data : pd.Series
        Time series of data to be smoothed. Must have a datetime index.
    **kwargs
        Additional keyword arguments to pass to the ccgFilter class. Its times
        are in days of 1/365 of a decimal year (a calendar day, or 366/365 of
        one in a leap year): the cutoffs ``shortterm`` (default 80) and
        ``longterm`` (default 667), and ``sampleinterval``, the step of the
        even grid that the filter interpolates the data onto. With the default,
        ``sampleinterval=0``, the filter uses the mean of the steps between
        samples that are about a day or more. When there are none (data more
        often than daily with no gap of a day, or daily data within one leap
        year), lair passes the median step between samples instead (1/24 for
        hourly data). For data more often than daily *with* gaps of a day or
        more, the filter's estimate is the mean length of those gaps. The
        filter interpolates onto its grid, it does not average, so a grid that
        coarse samples the data at about the same hour each time and a diurnal
        cycle shifts the curve. Pass the data's own step there
        (``sampleinterval=1/24`` for hourly data).

    Returns
    -------
    ccgFilter
        Thoning filter object.
    """
    if data.isna().any():
        raise ValueError("Data contains NaN values.")

    # Convert datetime index to decimal date
    xp = data.index.to_series().apply(dt2decimalDate).to_numpy(dtype=float)
    yp = data.values

    if kwargs.get("sampleinterval", 0) == 0:
        # ccgFilter estimates the interval from the steps longer than 0.002739
        # years (about a day) and divides by zero when there are none. Only then,
        # pass the median step, in the filter's unit (1/365 of a decimal year).
        steps = np.diff(np.sort(xp))
        if not (steps > 0.002739).any():
            steps = steps[steps > 0]
            if steps.size == 0:
                raise ValueError("Data needs samples at two or more distinct times.")
            kwargs["sampleinterval"] = float(np.median(steps)) * 365
            logger.debug("sampleinterval set to %g days", kwargs["sampleinterval"])

    if "debug" not in kwargs:
        # Print the filter's diagnostics when lair.background logs at DEBUG
        kwargs["debug"] = logger.isEnabledFor(logging.DEBUG)

    # Fit the Thoning curve
    return ccgFilter(xp, yp, **kwargs)


def thoning(
    data: pd.Series, smooth_time: list[dt.datetime] | pd.Index | None = None, **kwargs
) -> pd.Series:
    """
    Thoning curve fitting.

    Wraps code published by NOAA GML: https://gml.noaa.gov/ccgg/mbl/crvfit/crvfit.html

    Thoning, K.W., P.P. Tans, and W.D. Komhyr, 1989,
        Atmospheric carbon dioxide at Mauna Loa Observatory,
        2. Analysis of the NOAA/GMCC data, 1974 1985.,
        J. Geophys. Res. ,94, 8549 8565.

    Parameters
    ----------
    data : pd.Series
        Time series of data to be smoothed. Must have a datetime index.
        NaN values are dropped before the curve is fitted.
    smooth_time : list[datetime] | pd.Index, optional
        Times at which to evaluate the fitted curve; they become the index of
        the result. Default is the index of ``data``, so rows where ``data`` is
        NaN get the curve's value.
    **kwargs
        Additional keyword arguments to pass to the ccgFilter class (see
        :func:`thoning_filter`). ``sampleinterval`` is the step, in days, of
        the even grid the filter interpolates the data onto. The default (0)
        lets the filter estimate it from the steps of about a day or more;
        when the data has none, as hourly data without a gap of a day does,
        lair uses the median step (1/24 for hourly data). Hourly data with
        such gaps needs ``sampleinterval=1/24``.

    Returns
    -------
    pd.Series
        Smoothed data, at ``smooth_time``.
    """
    # Drop nans (filter does not handle them)
    orig_index = data.index.copy()  # however, we may want to return the original index
    data = data.dropna().sort_index()

    # Create a Thoning filter object
    filt = thoning_filter(data, **kwargs)

    # Get the times to return the smoothed data. Evaluate the curve at the
    # original times (including any NaN rows) rather than reusing filt.xp,
    # which is NaN-free and sorted, so it would not line up with orig_index
    if smooth_time is None:
        smooth_time = orig_index
    decimal_time = [dt2decimalDate(t) for t in smooth_time]

    # Return the smoothed data
    smooth = filt.getSmoothValue(decimal_time)
    return pd.Series(
        smooth, index=smooth_time, name=data.name
    )  # reassign index times to avoid issues with decimal rounding
