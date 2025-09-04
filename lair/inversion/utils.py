
import pandas as pd
import xarray as xr


def pandas_index_to_xarray_coords(index: pd.Index | pd.MultiIndex) -> xr.Coordinates:
    return index.to_series().to_xarray().coords


def integrate_over_time_bins(data: xr.DataArray, time_bins: pd.IntervalIndex,
                             time_dim: str = 'time') -> xr.DataArray:
    """
    Integrate data over time bins.

    Parameters
    ----------
    data : xr.DataArray
        Data to integrate.
    time_bins : pd.IntervalIndex
        Time bins for integration.
    time_dim : str, optional
        Time dimension name, by default 'time'

    Returns
    -------
    xr.DataArray
        Integrated footprint.
    """
    # Use pd.cut to bin the data by time into time bins
    integrated = data.groupby_bins(group=time_dim, bins=time_bins, include_lowest=True, right=False
                                   ).sum()

    # Fill nans with 0s
    integrated = integrated.fillna(0)
    return integrated


def get_dims_from_index(index: pd.Index) -> list[str]:
    """Get dimension names from a pandas Index or MultiIndex."""
    if isinstance(index, pd.MultiIndex):
        dims = list(index.names)
    elif index.name is not None:
        dims = [index.name]
    else:
        raise ValueError("Index must have a name or be a MultiIndex with names.")

    if any(dim is None for dim in dims):
        raise ValueError("Index must have names for all dimensions.")

    return dims


def get_index_from_coords(coords: xr.Coordinates, dims: str | list[str] | None = None) -> pd.Index:
    assert isinstance(coords, xr.Coordinates), "Coordinates must be an xarray Coordinates object."
    
    if dims is None:
        return coords.to_index()
    elif isinstance(dims, str):
        return coords[dims].to_index()
    elif isinstance(dims, list):
        if not all(dim in coords for dim in dims):
            raise ValueError(f"Not all dimensions {dims} are present in the coordinates.")
        return pd.MultiIndex.from_product(
            (coords[dim].values for dim in dims),
            names=dims
        )
    else:
        raise TypeError("dims must be a string, list of strings, or None.")


def merge_multiindexes(multiindexes: list[pd.MultiIndex]) -> pd.MultiIndex:
    """
    Merge multiple MultiIndexes into a single MultiIndex.

    Parameters
    ----------
    multiindexes : list[pd.MultiIndex]
        List of MultiIndexes to merge.

    Returns
    -------
    pd.MultiIndex
        Merged MultiIndex.
    """
    return pd.MultiIndex.from_product(
        [level for index in multiindexes for level in index.levels],
    )