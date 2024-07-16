"""
Calculate background concentrations.
"""

import datetime as dt
import pandas as pd

from lair.config import verbose
from lair.air._ccg_filter import ccgFilter
from lair.utils.clock import dt2decimalDate


def baseline(data: pd.Series, window: int=24, q: float=0.01
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


def thonning(data: pd.Series, return_filt: bool=False, **kwargs
             )-> ccgFilter | pd.Series:
    """
    Thonning curve fitting.

    Wraps code published by NOAA GML
        https://gml.noaa.gov/ccgg/mbl/crvfit/crvfit.html

    Thoning, K.W., P.P. Tans, and W.D. Komhyr, 1989,
        Atmospheric carbon dioxide at Mauna Loa Observatory,
        2. Analysis of the NOAA/GMCC data, 1974 1985.,
        J. Geophys. Res. ,94, 8549 8565.

    Parameters
    ----------
    data : pd.Series
        Time series of data to be smoothed. Must have a datetime index.
    return_filt : bool, optional
        Return the filter object instead of the smoothed data.
    **kwargs
        Additional keyword arguments to pass to the ccgFilter class.

    Returns
    -------
    ccgFilter | pd.Series
        If return_filt is True, returns the filter object.
        Otherwise, returns the smoothed data as a pandas series.
    """
    # Convert datetime index to decimal date
    data = data.dropna()
    xp = data.index.to_series().apply(dt2decimalDate).values
    yp = data.values

    # Fit the Thonning curve
    filt = ccgFilter(xp, yp, debug=verbose, **kwargs)
    if return_filt:
        return filt  # Return the filter object

    # Return the smoothed data
    smooth = filt.getSmoothValue(xp)
    return pd.Series(smooth, index=data.index)
