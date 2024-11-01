"""
Calculate background concentrations.
"""

import datetime as dt
import pandas as pd

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


def rolling_baseline(data: pd.Series, window: int=24, q: float=0.01
             ) -> pd.Series:
    """
    Calculate the baseline concentration as the {q} quantile of a rolling
    window of size {window} hours.

    Parameters
    ----------
    data : pd.Series
        Time series of data to calculate the baseline from.
    window : int, optional
        Size of the rolling window in hours.
    q : float, optional
        Quantile to calculate the baseline from.

    Returns
    -------
    pd.Series
        Baseline concentration
    """
    window = dt.timedelta(hours=window)
    baseline = (data.rolling(window=window, center=True).quantile(q)
                    .rolling(window=window, center=True).mean())

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


def thonning_filter(data: pd.Series, **kwargs) -> ccgFilter:
    """
    Create a Thonning filter object from a time series of data.

    Parameters
    ----------
    data : pd.Series
        Time series of data to be smoothed. Must have a datetime index.
    **kwargs
        Additional keyword arguments to pass to the ccgFilter class.

    Returns
    -------
    ccgFilter
        Thonning filter object.
    """
    # Convert datetime index to decimal date
    data = data.dropna()
    xp = data.index.to_series().apply(dt2decimalDate).values
    yp = data.values

    if not 'debug' in kwargs:
        # Set debug level using lair's verbose setting
        kwargs['debug'] = verbose

    # Fit the Thonning curve
    return ccgFilter(xp, yp, **kwargs)

def thonning(data: pd.Series,
             smooth_time: list[dt.datetime] | None = None,
             **kwargs
             )-> ccgFilter | pd.Series:
    """
    Thonning curve fitting.

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

    """
    # Create a Thonning filter object
    filt = thonning_filter(data, **kwargs)

    # Get the times to return the smoothed data
    if smooth_time is None:
        # Use the original time series
        smooth_time = data.index
        decimal_time = filt.xp
    else:
        decimal_time = [dt2decimalDate(t) for t in smooth_time]

    # Return the smoothed data
    smooth = filt.getSmoothValue(decimal_time)
    return pd.Series(smooth, index=smooth_time)  # reassign index times to avoid issues with decimal rounding
