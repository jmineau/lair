"""
Utility functions for inversion module.
"""

import pandas as pd
from pandas.api.types import is_float_dtype
import xarray as xr


def integrate_over_time_bins(data: pd.DataFrame | pd.Series, time_bins: pd.IntervalIndex,
                             time_dim: str = 'time') -> pd.DataFrame | pd.Series:
    """
    Integrate data over time bins.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        Data to integrate.
    time_bins : pd.IntervalIndex
        Time bins for integration.
    time_dim : str, optional
        Time dimension name, by default 'time'

    Returns
    -------
    pd.DataFrame | pd.Series
        Integrated footprint. The bin labels are set to the left edge of the bin.
    """
    is_series = isinstance(data, pd.Series)
    
    dims = data.index.names
    if time_dim not in dims:
        raise ValueError(f"time_dim '{time_dim}' not found in data index levels {dims}")
    other_levels = [lvl for lvl in dims if lvl != time_dim]

    data = data.reset_index()

    # Use pd.cut to bin the data by time into time bins
    data[time_dim] = pd.cut(data[time_dim], bins=time_bins,
                            include_lowest=True, right=False)

    # Set Intervals to the left edge of the bin (start of time interval)
    data[time_dim] = data[time_dim].apply(lambda x: x.left)

    # Group the date by the time bins & any other existing levels
    grouped = data.groupby([time_dim] + other_levels, observed=True)

    # Sum over the groups
    integrated = grouped.sum()

    # Order the index levels
    integrated = integrated.reorder_levels(dims)

    if is_series:
        # Return a Series if the input was a Series
        return integrated.iloc[:, 0]
    return integrated


def round_index(index: pd.Index | pd.MultiIndex, decimals: int
                ) -> pd.Index | pd.MultiIndex:
    """
    Rounds the values in a pandas Index or MultiIndex if the level's
    data type is a numpy floating type.

    Parameters
    ----------
    index : pd.Index | pd.MultiIndex
        Input index to round.
    decimals : int
        Number of decimal places to round to.

    Returns
    -------
    pd.Index | pd.MultiIndex
        Rounded index.
    """
    if not isinstance(index, (pd.Index, pd.MultiIndex)):
        raise TypeError("Input must be a pandas Index or MultiIndex.")

    if isinstance(index, pd.MultiIndex):
        # Handle MultiIndex
        new_levels = []
        changed = False
        for i in range(index.nlevels):
            level = index.levels[i]
            if is_float_dtype(level):
                # Round the level if it's a float type
                new_levels.append(level.round(decimals))
                changed = True
            else:
                new_levels.append(level)
        
        if changed:
            # Reconstruct the MultiIndex with the new, rounded levels
            return pd.MultiIndex.from_arrays(
                [index.get_level_values(i) for i in range(index.nlevels)],
                names=index.names
            ).set_levels(new_levels)
        else:
            # Return original index if no levels were changed
            return index

    elif is_float_dtype(index.dtype):
        # Handle single Index
        return index.round(decimals)
    else:
        # Return original index if it's not a float type
        return index


def dataframe_matrix_to_xarray(frame: pd.DataFrame) -> xr.DataArray:
    """
    Convert a pandas DataFrame to an xarray DataArray.

    If the DataFrame has a MultiIndex for columns, all levels of the MultiIndex
    are stacked into the index of the resulting DataArray.

    Parameters
    ----------
    frame : pd.DataFrame
        DataFrame to convert.

    Returns
    -------
    xr.DataArray
        Converted DataArray.
    """

    if isinstance(frame.columns, pd.MultiIndex):
        # Stack all levels of the columns MultiIndex into the index
        n_levels = len(frame.columns.levels)
        s = frame.stack(list(range(n_levels)), future_stack=True)
    else:
        s = frame.stack(future_stack=True)
    return s.to_xarray()
