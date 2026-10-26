"""
Background concentrations of a trace gas from its own time series.

:func:`rolling_baseline` and :func:`phase_shift_corrected_baseline` take a low
quantile of a moving window; :func:`thoning` fits NOAA's curve (Thoning et al.,
1989) through the data, which suits afternoon means (the afternoon is when the
boundary layer is deep; :data:`lair.clock.AFTERNOON` lists its hours).
CarbonTracker's background is in :mod:`lair.noaa`.
:doc:`/examples/background` shows the methods side by side and when to use
which. The module needs NOAA's CCG filter, which the first ``import lair``
downloads (see :ref:`first-import-download`), and ``scipy``.
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
from lair.clock import dt2decimalDate

logger = logging.getLogger(__name__)


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

    See Also
    --------
    phase_shift_corrected_baseline : A low quantile that follows fast changes
        in one-second data.
    thoning : A smooth curve through the data.

    Examples
    --------
    Hourly methane at 2000 ppb with a spike of 50 ppb every morning at 08:00;
    the baseline leaves the spikes out:

    >>> import pandas as pd
    >>> from lair.background import rolling_baseline
    >>> times = pd.date_range("2024-01-01", periods=72, freq="h")
    >>> ch4 = pd.Series(2000.0, index=times)
    >>> ch4[times.hour == 8] += 50.0
    >>> rolling_baseline(ch4, window="24h").unique()
    array([2000.])
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
        The windows are counted on a one-second grid that starts at each
        day's first time, so the times must fall on whole seconds from it
        (one-second data, or a slower rate on whole seconds).
    n : int, optional
        Window size in seconds. Default is 3600.
    q : float, optional
        Quantile to extract from signal. Default is 0.01.

    Returns
    -------
    pd.Series
        Baseline concentration at the times of ``data``, so ``data - baseline``
        lines up row for row. A time off the one-second grid gets NaN.

    See Also
    --------
    rolling_baseline : A rolling low quantile for data of any rate.

    Examples
    --------
    Two hours of one-second methane at 2000 ppb with a plume of 100 ppb for
    five minutes, and a window of ten minutes:

    >>> import pandas as pd
    >>> from lair.background import phase_shift_corrected_baseline
    >>> times = pd.date_range("2024-01-01 12:00", periods=7200, freq="s")
    >>> ch4 = pd.Series(2000.0, index=times)
    >>> ch4.iloc[3000:3300] += 100.0
    >>> baseline = phase_shift_corrected_baseline(ch4, n=600)
    >>> baseline.round(6).unique()
    array([2000.])
    >>> round(float((ch4 - baseline).max()), 6)  # the plume above the baseline
    100.0
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

    # the windows run on the one-second grid; give the baseline at the input's times
    return pd.concat(b).reindex(data.index)


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
        Thoning filter object. Its methods (``getSmoothValue``,
        ``getTrendValue``, ``getHarmonicValue``, ...) take decimal years
        (:func:`lair.clock.dt2decimalDate`).

    See Also
    --------
    thoning : The smooth curve at given times, from this filter.

    Examples
    --------
    The trend of three years of made-up daily CO2 that grows 2.5 ppm a year,
    with a seasonal cycle and noise:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from lair.background import thoning_filter
    >>> from lair.clock import dt2decimalDate
    >>> rng = np.random.default_rng(0)
    >>> days = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    >>> years = (days - days[0]).days / 365.25
    >>> seasons = 3 * np.cos(2 * np.pi * years)
    >>> noise = rng.normal(0, 0.5, days.size)
    >>> co2 = pd.Series(410 + 2.5 * years + seasons + noise, index=days)
    >>> filt = thoning_filter(co2)
    >>> jan1 = [dt2decimalDate(pd.Timestamp(f"{y}-01-01")) for y in (2021, 2022)]
    >>> filt.getTrendValue(np.array(jan1)).round(1)
    array([412.5, 415. ])
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

    See Also
    --------
    thoning_filter : The filter itself, for its trend and harmonics.
    rolling_baseline : A low quantile instead of a curve through everything.

    Examples
    --------
    Three years of made-up daily CO2 that grows 2.5 ppm a year, with a seasonal
    cycle and noise, smoothed, then read on the first of each month:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from lair.background import thoning
    >>> rng = np.random.default_rng(0)
    >>> days = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    >>> years = (days - days[0]).days / 365.25
    >>> seasons = 3 * np.cos(2 * np.pi * years)
    >>> noise = rng.normal(0, 0.5, days.size)
    >>> co2 = pd.Series(410 + 2.5 * years + seasons + noise, index=days)
    >>> smooth = thoning(co2)
    >>> round(float(smooth.iloc[-1] - smooth.iloc[0]), 1)
    7.6
    >>> months = pd.date_range("2020-01-01", "2022-12-01", freq="MS")
    >>> thoning(co2, smooth_time=months).round(1).head(3).tolist()
    [413.0, 412.8, 412.1]
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
