"""
Bayesian Flux Inversion
"""

import datetime as dt
import itertools
import numpy as np
import pandas as pd
from typing import Any
import xarray as xr

from lair.air.stilt import Receptor, Footprint, STILT
from .bayesian import Inversion


# TODO:
# - non lat-lon grids


def generate_indices(receptor_obs: dict[Receptor, pd.Series],
                     t_start: dt.datetime, t_end: dt.datetime,
                     freq: Any,
                     out_grid: xr.DataArray,
                     grid_mask: xr.DataArray | None = None):
    """
    Generate Pandas MultiIndex objects for observations and flux state

    Parameters
    ----------
    receptor_obs : dict[Receptor, pd.Series]
        Dictionary of receptor observations.
        Keys are Receptor objects and values are a list/series of datetime objects.
    t_start : dt.datetime
        Start time of the inversion period
    t_end : dt.datetime
        End time of the inversion period
    freq : Any
        Frequency of the inversion time steps. Passed to `pd.date_range`.
    out_grid : xr.DataArray
        Output grid for the inversion. Must have 'lon' and 'lat' dimensions.
    grid_mask : xr.DataArray, optional
        Mask for the output grid. Must have the same coordinates as `out_grid`.
        Grid cells where the mask == 0 will be excluded from the inversion.

    Returns
    -------
    obs_index : pd.MultiIndex
        MultiIndex for observations
    flux_index : pd.MultiIndex
        MultiIndex for flux state
    """
    # Generate observation index
    receptor_obs_tuples = [(r, t) for r, times in receptor_obs.items()
                           for t in times]
    obs_index = pd.MultiIndex.from_tuples(receptor_obs_tuples, names=['receptor', 'obs_time'])

    # Generate flux index
    flux_times = pd.date_range(t_start, t_end, freq=freq)
    x_dim, y_dim = 'lon', 'lat'
    x, y = out_grid[x_dim].values, out_grid[y_dim].values
    xx, yy = np.meshgrid(x, y, indexing='ij')
    x_y_tuples = pd.Index(zip(xx.ravel(), yy.ravel())).values.reshape(xx.shape)
    cells = xr.DataArray(x_y_tuples, coords={x_dim: x, y_dim: y})

    if grid_mask:
        # Mask grid cells where mask == 0
        cells = cells.where(grid_mask)

    flux_index = pd.MultiIndex.from_product([flux_times, cells.values.ravel()],
                                            names = ['flux_time', 'cell'])

    return obs_index, flux_index



class Jacobian:
    """
    Jacobian built from STILT footprints
    """

    def __init__(self,
                 footprints: xr.DataArray):
        # need footprints and receptors
        # need outgrid and out time steps
        # sparse coo foot matrix per inversion time step
        
        self.footprints = footprints