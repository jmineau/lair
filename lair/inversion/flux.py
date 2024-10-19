"""
Bayesian Flux Inversion
"""

import datetime as dt
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import scipy
import sparse
from typing import Any
import xarray as xr
import xesmf as xe

from lair.air.stilt import Receptor, Footprint
from lair.air.stilt import Model as STILTModel
from lair.inversion.bayesian import Inversion
from lair.inventories import Inventory
from lair.utils.clock import TimeRange


# TODO:
# - non lat-lon grids
# - multiple receptors
# - polygonal flux "cells"
# - inversion wkspace


def generate_obs_index(receptor_obs: dict['Receptor', pd.Series]) -> pd.MultiIndex:
    """
    Generate Pandas MultiIndex object for observations.

    Parameters
    ----------
    receptor_obs : dict[Receptor, pd.Series]
        Dictionary of receptor observations.
        Keys are Receptor objects and values are a list/series of datetime objects.

    Returns
    -------
    obs_index : pd.MultiIndex
        MultiIndex for observations
    """
    receptor_obs_tuples = [(r, t) for r, times in receptor_obs.items() for t in times]
    obs_index = pd.MultiIndex.from_tuples(receptor_obs_tuples, names=['receptor', 'obs_time'])
    return obs_index


def add_receptor_index(receptor: Receptor, obs: pd.Series):
    """
    Add receptor index to observations with DateTimeIndex
    by converting the index to a MultiIndex[receptor, obs_time].

    Parameters
    ----------
    receptor : Receptor
        Receptor object.
    obs : pd.Series
        Observations.

    Returns
    -------
    obs : pd.Series
        Observations with receptor-obs_time MultiIndex.
    """
    obs.index.name = 'obs_time'
    obs = obs.reset_index()
    obs['receptor'] = receptor
    obs = obs.set_index(['receptor', 'obs_time'])
    return obs


def generate_regular_flux_times(t_start: dt.datetime, t_end: dt.datetime, freq: str) -> pd.IntervalIndex:
    """
    Generate regular time bins for the inversion.

    Parameters
    ----------
    t_start : dt.datetime
        Start time for the inversion.
    t_end : dt.datetime
        End time for the inversion.
    freq : str
        Frequency for the time bins.

    Returns
    -------
    flux_times : pd.IntervalIndex
        Regular time bins for the inversion.
    """
    flux_times = pd.interval_range(start=t_start, end=t_end, freq=freq)
    return flux_times


def generate_flux_index(t_bins: pd.IntervalIndex,
                        out_grid: xr.DataArray, grid_mask: xr.DataArray | None = None) -> pd.MultiIndex:
    """
    Generate Pandas MultiIndex object for flux state.

    Parameters
    ----------
    t_bins : pd.IntervalIndex
        Time bins for the inversion.
    out_grid : xr.DataArray
        Output grid for the inversion. Must have 'lon' and 'lat' dimensions.
    grid_mask : xr.DataArray, optional
        Mask for the output grid. Must have the same coordinates as `out_grid`.
        Grid cells where the mask == 0 will be excluded from the inversion.

    Returns
    -------
    flux_index : pd.MultiIndex
        MultiIndex for flux state
    """

    x_dim, y_dim = 'lon', 'lat'
    x, y = out_grid[x_dim].values, out_grid[y_dim].values
    xx, yy = np.meshgrid(x, y, indexing='ij')
    x_y_tuples = pd.Index(zip(xx.ravel(), yy.ravel())).values.reshape(xx.shape)
    cells = xr.DataArray(x_y_tuples, coords={x_dim: x, y_dim: y})

    if grid_mask is not None:
        # Mask grid cells where mask == 0
        cells = cells.where(grid_mask)

    flux_index = pd.MultiIndex.from_product([t_bins, cells.values.ravel()],
                                            names=['flux_time', 'cell'])
    return flux_index

def generate_state(t_bins: pd.IntervalIndex,
                   out_grid: xr.DataArray, grid_mask: xr.DataArray | None = None) -> xr.DataArray:
    """
    Generate state matrix for the inversion.

    Parameters
    ----------
    t_bins : pd.IntervalIndex
        Time bins for the inversion.
    out_grid : xr.DataArray
        Output grid for the inversion. Must have 'lon' and 'lat' dimensions.
    grid_mask : xr.DataArray, optional
        Mask for the output grid. Must have the same coordinates as `out_grid`.
        Grid cells where the mask == 0 will be excluded from the inversion.

    Returns
    -------
    state : xr.DataArray
        State matrix for the inversion.
    """
    flux_index = generate_flux_index(t_bins, out_grid, grid_mask)
    state = xr.DataArray(np.zeros(len(flux_index)), coords=[flux_index], name='state')
    return state


class FluxInversion(Inversion):
    """
    Bayesian flux inversion
    """
    def __init__(self,
                 project: str | Path,
                 obs: pd.Series,  # with MultiIndex[receptor, obs_time]
                 background: pd.Series,  # with Index[obs_time]
                 inventory: Inventory,
                 stilt: STILTModel,
                 t_bins: pd.IntervalIndex,
                 out_grid: xr.DataArray,
                 grid_buffer: float = 0.1,
                 grid_mask: xr.DataArray | None = None,
                 bio: xr.DataArray | None = None,
                 prior_error_cov: xr.DataArray | None = None,
                 modeldata_mismatch: None = None,
                 write_jacobian: bool = True,
                 regrid_weights: str | Path | None = None
                 ) -> None:
        self.project = Path(project)
        self.project.mkdir(exist_ok=True, parents=True)  # create project directory if it doesn't exist

        self.obs = obs
        self.background = background
        self.inventory = inventory
        self.stilt = stilt if isinstance(stilt, STILTModel) else STILTModel(stilt_wd=stilt)
        self.out_grid = out_grid
        self.grid_buffer = grid_buffer
        self.grid_mask = grid_mask
        self.bio = bio
        self.prior_error_cov = prior_error_cov
        self.modeldata_mismatch = modeldata_mismatch
        self.write_jacobian = write_jacobian
        self.regrid_weights = regrid_weights

        self.obs_index = obs.index
        self.flux_index = generate_flux_index(t_bins=t_bins, out_grid=out_grid, grid_mask=grid_mask)

        self.receptors = obs.index.get_level_values('receptor').unique()
        self.obs_times = obs.index.get_level_values('obs_time').unique()
        self.flux_times = t_bins
        self.cells = self.flux_index.get_level_values('cell').unique()

        x_0 = self.build_x_0(inventory)
        self.jacobian = self.build_jacobian(write_to_disk=write_jacobian, weights=regrid_weights)

        super().__init__(z=obs, c=background, x_0=x_0, H=self.jacobian,
                         S_0=prior_error_cov, S_z=modeldata_mismatch)
    
    def build_x_0(self, inventory: xr.DataArray) -> xr.DataArray:
        """
        Build the initial state matrix for the inversion.

        Parameters
        ----------
        inventory : Inventory
            Inventory object.

        Returns
        -------
        x_0 : xr.DataArray
            Initial state matrix for the inversion.
        """
        return inventory.groupby_bins(group='time', bins=self.flux_times).sum()

    def build_S_s(self, variance, length_scale):
        """
        Build the spatial error covariance matrix for the inversion.

        Parameters
        ----------
        variance : float
            Variance of the spatial error.
        length_scale : float
            Length scale of the spatial error.

        Returns
        -------
        S_s : xr.DataArray
            Spatial error covariance matrix for the inversion.
        """
        # Calculate the spatial error covariance matrix
        
        # TODO: from copilot - check if this is correct
        S_s = xr.DataArray(
            data=scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(self.out_grid[['lon', 'lat']].values, metric='euclidean')
            ),
            dims=['cell', 'cell'],
            coords={'cell': self.cells}
        )
        S_s = variance * np.exp(-S_s / length_scale)

        return S_s

    def build_S_t(self, variance, time_scale):
        """
        Build the temporal error covariance matrix for the inversion.

        Parameters
        ----------
        variance : float
            Variance of the temporal error.
        time_scale : float
            Time scale of the temporal error.

        Returns
        -------
        S_t : xr.DataArray
            Temporal error covariance matrix for the inversion.
        """
        # Calculate the temporal error covariance matrix
        
        # TODO: from copilot - check if this is correct
        S_t = xr.DataArray(
            data=scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(self.flux_times.mid.values.reshape(-1, 1), metric='euclidean')
            ),
            dims=['flux_time', 'flux_time'],
            coords={'flux_time': self.flux_times}
        )
        S_t = variance * np.exp(-S_t / time_scale)

        return S_t

    def build_S_0(self, prior_error_variances: xr.DataArray) -> xr.DataArray:
        """
        Build the prior error covariance matrix for the inversion.

        Parameters
        ----------
        prior_error_variances : xr.DataArray
            Prior error variances for the inversion.

        Returns
        -------
        S_0 : xr.DataArray
            Prior error covariance matrix for the inversion.
        """
        
        return prior_error_variances

    def build_S_z(self,):
        """
        Build the model-data mismatch covariance matrix for the inversion.

        Returns
        -------
        S_z : xr.DataArray
            Model-data mismatch covariance matrix for the inversion.
        """
        pass

    def build_jacobian(self, write_to_disk: bool = False, weights: str | Path | None = None) -> xr.DataArray:
        """
        Build the Jacobian matrix for the inversion from STILT footprints.

        Parameters
        ----------
        write_to_disk : bool, optional
            Write the Jacobian matrix to disk, by default False
        weights : str | Path | None, optional
            Path to regridding weights, by default None

        Returns
        -------
        H : xr.DataArray
            Jacobian matrix for the inversion.
        """

        H_rows = []
        for sim in self.stilt.simulations.values():
            if not sim.has_footprint:
                continue

            # First spatially clip the footprints to slightly larger than the output grid
            footprint = sim.footprint.clip_to_grid(self.out_grid, buffer=self.grid_buffer)
            del sim.footprint  # free up memory

            # Sum the footprints over the inversion time bins
            foot = footprint.integrate_over_time_bins(t_bins=self.flux_times)\
                .rename({'time_bins': 'flux_time',
                        'run_time': 'obs_time'})

            # Regrid the footprints to the output grid using conservative regridding
            weights = weights or self.regrid_weights
            regridder = xe.Regridder(foot.data, self.out_grid, method='conservative',
                                     filename=weights)#, parallel=True if not weights else False)
            # Adding parallel=True is complicated - xesmf doesn't like how my datasets are chunked?
            #  When I do chunk the out_grid, I get:
            #    ValueError: zip() argument 2 is shorter than argument 1
            #  When I don't chunk the out_grid, I get:
            #    ValueError: Using `parallel=True` requires the output grid to have chunks along all spatial dimensions.
            #    If the dataset has no variables, consider adding an all-True spatial mask with appropriate chunks.

            if not self.regrid_weights:
                # Write weights to disk
                self.regrid_weights = regridder.to_netcdf(filename=weights)

            regridded = regridder(foot.data, keep_attrs=True)

            # Stack the lon/lat dims into a single cell dim
            row = regridded.stack(cell=('lon', 'lat'))

            H_rows.append(row)

        H = xr.concat(H_rows, dim='receptor')  # should probably be able to concat over receptor or obs_time

        # Check for empty rows
        empty_rows = H.isnull().all(dim=['flux_time', 'cell'])
        if empty_rows.any():
            print(f'Warning: Empty rows in Jacobian matrix for receptors: {empty_rows}')
        else:
            print('Jacobian matrix built successfully.')

        if write_to_disk:
            # Write to disk
            H.to_netcdf(self.stilt.stilt_wd / 'jacobian.nc')

        return H

    @staticmethod
    def _preprocess_grid(grid: xr.DataArray | xr.Dataset):
        pass