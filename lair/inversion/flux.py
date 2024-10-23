"""
Bayesian Flux Inversion
"""

import datetime as dt
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import sparse
import xarray as xr
import xesmf as xe
from lair import uataq
from lair.air.stilt import Footprint
from lair.air.stilt import Model as STILTModel
from lair.air.stilt import Receptor
from lair.inventories import Inventory
from lair.inversion.bayesian import Inversion
from lair.utils.clock import TimeRange
from lair.utils.geo import earth_radius
from sklearn.metrics.pairwise import haversine_distances

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


def add_receptor_index(receptor: Receptor, obs: pd.Series) -> pd.Series:
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
    obs = obs.reset_index().assign(receptor=receptor)  # add receptor column
    obs = obs.set_index(['receptor', 'obs_time'])  # set MultiIndex
    return obs


def uataq_receptor_obs(SID: str, method: Literal['get_obs', 'read_data'] = 'get_obs', **kwargs):
    """
    Get observations for UATAQ receptors.
    
    Parameters
    ----------
    SID : str
        Site ID
    method : 'get_obs' | 'read_data', optional
        UATAQ method to get observations, by default 'get_obs'.
    **kwargs
        Additional keyword arguments to pass to the UATAQ method

    Returns
    -------
    receptor_obs : pd.DataFrame
        Receptor observations with a MultiIndex[receptor, obs_time]
    """
    # Build site and receptor objects
    site = uataq.get_site(SID)  # get Site object
    latitude = site.config['latitude']
    longitude = site.config['longitude']
    zagl = site.config['zagl']
    receptor = Receptor(latitude=latitude, longitude=longitude, height=zagl)

    # Get observations
    data = getattr(site, method)(**kwargs)
    if method == 'read_data':
        # Need instrument to get data from read_data dict
        instrument = kwargs['instrument']
        data = data[instrument]

    return add_receptor_index(receptor, data)


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
    y_x_tuples = pd.Index(zip(yy.ravel(), xx.ravel())).values.reshape(yy.shape)
    cells = xr.DataArray(y_x_tuples, coords={y_dim: y, x_dim: x})

    if grid_mask is not None:
        # Mask grid cells where mask == 0
        cells = cells.where(grid_mask)

    flux_index = pd.MultiIndex.from_product([t_bins, cells.values.ravel()],
                                            names=['flux_time', 'cell'])
    return flux_index


def build_spatial_corr(cells: xr.DataArray, method: str | dict | None = None, **kwargs) -> np.ndarray:
    """
    Build the spatial error correlation matrix for the inversion.
    Has dimensions of cells x cells.

    Parameters
    ----------
    cells : xr.DataArray
        Flux cells.
    method : str, dict, optional
        Method for calculating the spatial error correlation matrix.
        Options include:
            - 'exp': exponentially decaying with an e-folding length kwargs['length_scale']
            - 'downscale': each cell is highly correlated with the same cell in the kwargs['src_grid'].
                            Correlation is defined as the multiplication of the `xesmf.Regridder` weights
                            between dst cells.
        Defaults to 'exp'.

    Returns
    -------
    S_s : np.ndarray
        Spatial error correlation matrix for the inversion.
    """
    # Handle method input
    if method is None:
        method = {'exp': 1.0}
    elif isinstance(method, str):
        method = {method: 1.0}
    elif isinstance(method, dict):
        assert sum(method.values()) == 1.0, 'Weights for spatial error methods must sum to 1.0'
    else:
        raise ValueError('method must be a string or dict')

    # Get the lat/lon coordinates of the flux cells
    cells = np.array([*cells.to_numpy()])  # convert to 2D numpy array [[lat, lon], ...]
    N = len(cells)  # number of flux cells

    # Initialize the spatial correlation matrix
    corr = np.zeros((N, N))

    # Calculate and combine correlation matrices based on the method weights
    for method_name, weight in method.items():
        if method_name == 'exp':  # Closer cells are more correlated
            distances = haversine_distances(np.deg2rad(cells)) * earth_radius(cells[:, 0])
            length_scale = kwargs['length_scale']
            method_corr = np.exp((-1 * distances) / length_scale)

        elif method_name == 'downscale':  # Cells that are from the same source cell are highly correlated

            src_grid = kwargs['src_grid']  # TODO: would prefer to get this from regridder or something, but not sure how to do that right now
            regridder = kwargs['regridder']

            # Reshape the weights to the shape of the regridder and assign coords
            # https://github.com/pangeo-data/xESMF/blob/7083733c7801960c84bf06c25ca952d3b44eac3e/xesmf/frontend.py#L596
            weights = regridder.weights.data.reshape(regridder.shape_out + regridder.shape_in)
            weights = xr.DataArray(weights, coords={
                'lat': cells[:, 0],
                'lon': cells[:, 1],
                'src_lat': src_grid['lat'],
                'src_lon': src_grid['lon'],
            }).stack(cell=('lat', 'lon'), src_cell=('src_lat', 'src_lon'))

            src_cells = weights.coords['src_cell'].values  # stack source cell coords

            # Create an empty matrix with dimensions len(cells) x len(cells)
            method_corr = np.zeros((N, N))

            # Set the diagonal elements to 1
            np.fill_diagonal(method_corr, 1)

            # Populate the matrix using broadcasting
            for in_cell in src_cells:
                # For each input cell, find the output cells that share the same input cell
                in_cell_weights = weights.sel(cell_in=in_cell)
                shared_out_cells = np.where((in_cell_weights > 0).values)[0]
                if len(shared_out_cells) > 0:
                    # Get the weight values for the shared output cells
                    values = weights.sel(in_cell=in_cell).values[shared_out_cells]
                    # Set the correlation matrix values for the shared output cells
                    # (np.ix_ is a fancy way to symmetrically index a 2D array)
                    method_corr[np.ix_(shared_out_cells, shared_out_cells)] = np.outer(values, values)

        else:
            raise ValueError(f"Unknown method: {method_name}")

        corr += weight * method_corr

    return corr


def build_temporal_corr(times: pd.DatetimeIndex, method: str | dict | None = None, **kwargs) -> np.ndarray:
    """
    Build the temporal error correlation matrix for the inversion.
    Has dimensions of flux_times x flux_times.

    Parameters
    ----------
    times : pd.DatetimeIndex
    method : dict, optional
        Method for calculating the temporal error correlation matrix.
        The key defines the method and the value is the weight for the method.
        Options include:
            - 'exp': exponentially decaying with an e-folding length of self.t_bins freq
            - 'diel': like 'exp', except correlations are only non-zero for the same time of day
            - 'clim': each month is highly correlated with the same month in other years

    Returns
    -------
    S_t : xr.DataArray
        Temporal error correlation matrix for the inversion.
    """
    # Handle method input
    if method is None:
        method = {'exp': 1.0}
    elif isinstance(method, str):
        method = {method: 1.0}
    elif isinstance(method, dict):
        assert sum(method.values()) == 1.0, 'Weights for temporal error methods must sum to 1.0'
    else:
        raise ValueError('method must be a dict')

    # Initialize the temporal correlation matrix
    N = len(times)  # number of flux times
    corr = np.zeros((N, N))

    # Calculate and combine correlation matrices based on the method weights
    for method_name, weight in method.items():
        if method_name in ['exp', 'diel']:  # Closer times are more correlated
            # Calculate the time differences between the midpoint of the flux times
            time_diffs = np.abs(np.subtract.outer(times, times))

            # Wrap in pandas DataFrame to use pd.Timedelta functionality
            time_diffs = pd.DataFrame(time_diffs)

            # Get time_scale as a pd.Timedelta
            # time_scale is formatted as a string like '1D' or '1h'
            time_scale = pd.Timedelta(kwargs['time_scale'])

            # Calculate the correlation matrix using an exponential decay
            method_corr =  np.exp(-time_diffs / time_scale).values  # values gets the numpy array

            if method_name == 'diel':
                # Set the correlation values for the same hour of day
                hours = times.hour
                same_time_mask = (hours[:, None] - hours[None, :]) == 0
                method_corr[~same_time_mask] = 0

        elif method_name == 'clim':  # Each month is highly correlated with the same month in other years
            # Initialize the correlation matrix as identity matrix
            method_corr = np.eye(N)  # the diagonal

            # Set the correlation values for the same month in other years
            corr_val = 0.9
            months = times.month.values

            # Create a mask for the same month in different years
            same_month_mask = (months[:, None] - months[None, :]) % 12 == 0

            # Apply the correlation value using the mask
            method_corr[same_month_mask] = corr_val
        else:
            raise ValueError(f"Unknown method: {method_name}")

        corr += weight * method_corr

    return corr


class FluxInversion(Inversion):
    """
    Bayesian flux inversion
    """

    _reuse_options = ['prior', 'prior_error', 'prior_error_cov', 'jacobian']

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
                 regrid_weights: str | Path | None = None,
                 reuse: bool | str | list[str] = True,
                 ) -> None:
        # Set project directory
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
        self.reuse = reuse

        self.obs_index = obs.index  # FIXME: should be generate_obs_index(receptor_obs)
        self.flux_index = generate_flux_index(t_bins=t_bins, out_grid=out_grid, grid_mask=grid_mask)

        self.receptors = self.obs_index.get_level_values('receptor').unique()
        self.obs_times = self.obs_index.get_level_values('obs_time').unique()
        self.flux_times = self.flux_index.get_level_values('flux_time').unique()
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

    def build_S_0(self, prior_error_variances: xr.DataArray,
                  spatial_corr: dict | np.ndarray | None = None,
                  temporal_corr: dict | np.ndarray | None = None) -> xr.DataArray:
        """
        Build the prior error covariance matrix for the inversion.

        Parameters
        ----------
        prior_error_variances : xr.DataArray
            Prior error variances for the inversion.
        spatial_corr : dict | np.ndarray, optional
            Spatial error correlation matrix for the inversion, by default None.
            If None, the spatial error correlation matrix is an identity matrix.
            See `build_spatial_corr` for more options.
        temporal_corr : dict | np.ndarray, optional
            Temporal error correlation matrix for the inversion, by default None.
            If None, the temporal error correlation matrix is an identity matrix.
            See `build_temporal_corr` for more options.

        Returns
        -------
        S_0 : xr.DataArray
            Prior error covariance matrix for the inversion.
        """
        # If correlation matrices are None, set them to identity matrices
        # If correlation matrices are dicts, build them
        # If correlation matrices are np.ndarrays, use them

        if spatial_corr is None:
            spatial_corr = np.eye(len(self.cells))
        elif isinstance(spatial_corr, dict):
            spatial_corr = build_spatial_corr(cells=self.cells, **spatial_corr)
        elif not isinstance(spatial_corr, np.ndarray):
            raise ValueError('spatial_corr must be a dict or np.ndarray')

        if temporal_corr is None:
            temporal_corr = np.eye(len(self.flux_times))
        elif isinstance(temporal_corr, dict):
            temporal_corr = build_temporal_corr(times=self.flux_times.mid, **temporal_corr)
        elif not isinstance(temporal_corr, np.ndarray):
            raise ValueError('temporal_corr must be a dict or np.ndarray')

        # Build the prior error covariance matrix
        S_0 = (prior_error_variances ** 2) * np.kron(spatial_corr, temporal_corr)
        S_0 = xr.DataArray(S_0, coords=[self.flux_index, self.flux_index], name='S_0')

        return S_0

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
            H.to_netcdf(self.project / 'H' / 'jacobian.nc')

        return H

    @staticmethod
    def _preprocess_grid(grid: xr.DataArray | xr.Dataset):
        pass