"""
Calculate background concentrations.
"""

import datetime as dt
import pandas as pd
from typing import Any

from lair.config import verbose
from lair.air._ccg_filter import ccgFilter  # make available to user
from lair.utils.clock import AFTERNOON, dt2decimalDate, decimalDate2dt


def get_well_mixed(data: pd.Series | pd.DataFrame, hours: list[int]=AFTERNOON) -> pd.Series | pd.DataFrame:
    """
    Subset the data to the well-mixed hours of the day.

    Parameters
    ----------
    data : pd.Series | pd.DataFrame
        Time series data to subset. Must have a datetime index.
    hours : list[int], optional
        Hours of the day to subset. Default is LST afternoon hours.

    Returns
    -------
    pd.Series | pd.DataFrame
        Subset of the data for the well-mixed hours.
    """
    return data[data.index.hour.isin(hours)].resample('1d').mean()


def rolling_baseline(data: pd.Series, window: Any='24h', q: float=0.01,
                     min_periods: int=1, center: bool=True
             ) -> pd.Series:
    """
    Calculate the baseline concentration as the {q} quantile of a rolling
    window of size {window} hours.

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
    baseline = (data.rolling(window=window, center=center, min_periods=min_periods).quantile(q)
                    .rolling(window=window, center=center, min_periods=min_periods).mean())

    return baseline


def phase_shift_corrected_baseline(data: pd.Series, n: int = 3600, q: float = 0.01) -> pd.Series:
    """
    Derive a baseline concentration using a low quantile approach to minimize
    phase shift effects. This method uses forward-looking and backward-looking
    windows to better represent the lowest observed concentrations during
    periods of rapid change.

    .. note::
        Original developed by Ben Fasoli for the Google Street View project.
        See supplementary material in: https://doi.org/10.1016/j.atmosenv.2023.119995

    Parameters
    ----------
    data : pd.Series
        Signal time series with local datetime index.
    n : int
        Window size in seconds.
    q : float
        Quantile to extract from signal.
    timezone : str, optional
        Timezone to convert the datetime index to.

    Returns
    -------
    pd.Series
        Baseline concentration
    """
    if not n % 2:
        n += 1

    b = []
    for index, y in data.groupby(data.index.floor('d')):
        hz = y.asfreq('s')
        left = hz.rolling(n, min_periods=1).quantile(q)
        right = hz.iloc[::-1].rolling(n, min_periods=1).quantile(q).iloc[::-1]

        forward = right.rolling(n, min_periods=1).max()
        backward = left.iloc[::-1].rolling(n, min_periods=1).max().iloc[::-1]

        b.append(pd.concat([forward, backward], axis=1)
                 .max(axis=1)
                 .rolling(n, min_periods=1, center=True, win_type='blackman')
                 .mean())

    return pd.concat(b)


def thoning_filter(data: pd.Series, **kwargs) -> ccgFilter:
    """
    Create a Thoning filter object from a time series of data.

    Parameters
    ----------
    data : pd.Series
        Time series of data to be smoothed. Must have a datetime index.
    **kwargs
        Additional keyword arguments to pass to the ccgFilter class.

    Returns
    -------
    ccgFilter
        Thoning filter object.
    """
    if data.isna().any():
        raise ValueError("Data contains NaN values.")

    # Convert datetime index to decimal date
    xp = data.index.to_series().apply(dt2decimalDate).values
    yp = data.values

    if not 'debug' in kwargs:
        # Set debug level using lair's verbose setting
        kwargs['debug'] = verbose

    # Fit the Thoning curve
    return ccgFilter(xp, yp, **kwargs)

def thoning(data: pd.Series,
            smooth_time: list[dt.datetime] | None = None,
            **kwargs
            )-> pd.Series:
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
    
    **kwargs
        Additional keyword arguments to pass to the ccgFilter class.

    Returns
    -------
    pd.Series
        Smoothed data.
    """
    # Drop nans (filter does not handle them)
    orig_index = data.index.copy()  # however, we may want to return the original index
    data = data.dropna()

    # Create a Thoning filter object
    filt = thoning_filter(data, **kwargs)

    # Get the times to return the smoothed data
    if smooth_time is None:
        # Use the original time series
        smooth_time = orig_index
        decimal_time = filt.xp
    else:
        decimal_time = [dt2decimalDate(t) for t in smooth_time]

    # Return the smoothed data
    smooth = filt.getSmoothValue(decimal_time)
    return pd.Series(smooth, index=smooth_time)  # reassign index times to avoid issues with decimal rounding
