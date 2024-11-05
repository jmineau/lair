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
    receptor_obs_tuples = [(r.id, t) for r, times in receptor_obs.items() for t in times]
    obs_index = pd.MultiIndex.from_tuples(receptor_obs_tuples, names=['receptor', 'obs_time'])
    return obs_index


def add_receptor_index(receptor: Receptor, obs: pd.Series) -> pd.Series:
    """
    Add receptor index to observations with DateTimeIndex
    by converting the index to a MultiIndex[receptor.id, obs_time].

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
    obs = obs.reset_index().assign(receptor=receptor.id)  # add receptor column
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


def generate_regular_flux_times(t_start: dt.datetime, t_end: dt.datetime, freq: str,
                                closed: str = 'left') -> pd.IntervalIndex:
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
    flux_times = pd.interval_range(start=t_start, end=t_end, freq=freq, closed=closed)
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
    xx, yy = np.meshgrid(x, y)
    y_x_tuples = pd.Index(zip(yy.ravel(), xx.ravel())).values.reshape(xx.shape)
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
            - 'downscale': each cell is highly correlated with the same cell in the kwargs['coarse_grid'].
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

            coarse_grid = kwargs['coarse_grid']  # TODO: would prefer to get this from regridder or something, but not sure how to do that right now
            regridder = kwargs['regridder']

            # Reshape the weights to the shape of the regridder and assign coords
            # https://github.com/pangeo-data/xESMF/blob/7083733c7801960c84bf06c25ca952d3b44eac3e/xesmf/frontend.py#L596
            weights = regridder.weights.data.reshape(regridder.shape_out + regridder.shape_in)
            weights = xr.DataArray(weights.todense(),  # load sparse matrix to dense
                                   coords={
                                       'lat': regridder.out_coords['lat'].values,
                                       'lon': regridder.out_coords['lon'].values,
                                       'coarse_lat': coarse_grid['lat'].values,
                                       'coarse_lon': coarse_grid['lon'].values,},
                                   dims=['lat', 'lon',
                                         'coarse_lat', 'coarse_lon']
                                   ).stack(cell=('lat', 'lon'),
                                           coarse_cell=('coarse_lat', 'coarse_lon'))

            coarse_cells = weights.coords['coarse_cell'].values  # stack source cell coords

            # Create an empty matrix with dimensions len(cells) x len(cells)
            method_corr = np.zeros((N, N))

            # Populate the matrix using broadcasting
            for coarse_cell in coarse_cells:
                # For each input cell, find the output cells that share the same input cell
                coarse_cell_weights = weights.sel(coarse_cell=coarse_cell)
                shared_out_cells = np.where((coarse_cell_weights > 0).values)[0]
                if len(shared_out_cells) > 0:
                    # Get the weight values for the shared output cells
                    values = coarse_cell_weights.values[shared_out_cells]
                    # Set the correlation matrix values for the shared output cells as the outer product of the weights
                    # (np.ix_ is a fancy way to symmetrically index a 2D array)
                    method_corr[np.ix_(shared_out_cells, shared_out_cells)] = np.outer(values, values)

            # Overwrite the diagonal elements with 1
            np.fill_diagonal(method_corr, 1)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        # Combine the correlation matrices based on the method weights
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

        # Combine the correlation matrices based on the method weights
        corr += weight * method_corr

    return corr


def build_prior_error_cov(prior_error: xr.DataArray,
                          spatial_corr: np.ndarray, temporal_corr: np.ndarray
                          ) -> np.ndarray:
    """
    Build the prior error covariance matrix for the inversion.
    
    I am a little confused about what is supposed to be used as the prior error.
    Ultimately I know we want variances, but it seems like sometimes values are scaled
    or we set this value to an 
    """
    pass


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
    return data.groupby_bins(group=time_dim, bins=time_bins, include_lowest=True, right=False).sum()


class FluxInversion(Inversion):
    """
    Bayesian flux inversion
    """

    _reuse_options = ['prior', 'prior_error', 'prior_error_cov', 'jacobian']

    def __init__(self,
                 project: str | Path,
                 obs: pd.Series,  # with MultiIndex[receptor.id, obs_time]
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
                 reuse: bool | str | list[str] = True,
                 ) -> None:
        # Set project directory
        self.project = Path(project)
        self.project.mkdir(exist_ok=True, parents=True)  # create project directory if it doesn't exist

        self.obs = obs
        self.background = background
        self.inventory = inventory
        self.stilt = stilt if isinstance(stilt, STILTModel) else STILTModel(path=stilt)
        self.out_grid = out_grid
        self.grid_buffer = grid_buffer
        self.grid_mask = grid_mask
        self.bio = bio
        self.prior_error_cov = prior_error_cov
        self.modeldata_mismatch = modeldata_mismatch

        # Initialize regrid weights
        self.regrid_weights = {}
        self._regrid_path = self.project / 'regrid_weights'
        if self._regrid_path.exists():
            self.regrid_weights = {p.stem: p for p in self._regrid_path.iterdir()}
        else:
            self._regrid_path.mkdir(exist_ok=True, parents=True)

        # Reuse options
        if isinstance(reuse, str):
            reuse = [reuse]
        elif not isinstance(reuse, bool):
            reuse = [s.lower() for s in reuse]
        self.reuse = reuse

        # Generate indices
        self.obs_index = obs.index
        self.flux_index = generate_flux_index(t_bins=t_bins, out_grid=out_grid, grid_mask=grid_mask)

        self.receptors = self.obs_index.get_level_values('receptor').unique()
        self.obs_times = self.obs_index.get_level_values('obs_time').unique()
        self.flux_times = self.flux_index.get_level_values('flux_time').unique()
        self.cells = self.flux_index.get_level_values('cell').unique()

        # Build the prior state matrix
        x_0 = self.build_x_0(inventory)

        # Build the jacobian matrix from STILT footprints
        # self.jacobian = self.build_jacobian()

        # super().__init__(z=obs, c=background, x_0=x_0, H=self.jacobian,
        #                  S_0=prior_error_cov, S_z=modeldata_mismatch)

    def _should_build(self, objs: str | list[str], path: str | Path | None = None) -> bool:
        """
        Check if objects should be built based on the reuse attribute.

        Parameters
        ----------
        objs : str | list[str]
            Object names to check if it should be built.
        path : str | Path | None, optional
            Path to the object, by default None

        Returns
        -------
        bool
            True if the object should be built, False otherwise.
        """
        if not self.reuse:
            # If reuse is False, always build
            return True

        if path:
            path = Path(path)
            if path.is_dir():
                if not any(path.iterdir()):
                    # If the directory is empty, build
                    return True
            elif not path.exists():
                # If the path doesn't exist, build
                return True

        # Check if any of the objects are in the reuse list
        if isinstance(objs, str):
            objs = [objs]
        return any(obj.lower() in self.reuse for obj in objs)

    def build_x_0(self, inventory: xr.DataArray) -> xr.DataArray:
        """
        Build the prior state matrix for the inversion.

        Parameters
        ----------
        inventory : xarray.DataArray
            Inventory prior for the inversion.

        Returns
        -------
        x_0 : xr.DataArray
            Initial state matrix for the inversion.
        """
        return integrate_over_time_bins(data=inventory, time_bins=self.flux_times)

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

    def build_jacobian(self) -> xr.DataArray:
        """
        Build the Jacobian matrix for the inversion from STILT footprints.

        Returns
        -------
        H : xr.DataArray
            Jacobian matrix for the inversion.
        """
        path = self.project / 'H'

        if not self._should_build(objs=['jacobian', 'H'], path=path):
            print('Jacobian matrix already exists. Loading from disk.')
            return xr.open_mfdataset(path / 'jacobian_*.nc')

        print('Building Jacobian matrix...')

        # Get start and end times for the inversion
        t_start, t_stop = self.flux_times[0].left, self.flux_times[-1].right

        # Get successful simulations within the inversion time range
        print('Getting simulations within inversion time range...')
        assert self.stilt.n_hours < 0, 'STILT must be ran backwards in time'
        sim_df = self.stilt.simulations.to_dataframe(subset='successful').reset_index()
        assert len(sim_df) > 0, 'No successful simulations found'
        # TODO: perhaps can use the time_range attribute of the Simulation object, but would need to be able to pass additional columns in to_dataframe
        sim_df['sim_t_start'] = sim_df.run_time + pd.Timedelta(hours=self.stilt.n_hours)
        sim_df['sim_t_end'] = sim_df.run_time - pd.Timedelta(hours=1)  # subtract 1 hour to get the start of the end hour
        sim_df = sim_df[(sim_df.sim_t_start < t_stop) & (sim_df.sim_t_end >= t_start)]  # identify overlaps

        H_rows = []
        for sim_id in sim_df['sim_id']:
            sim = self.stilt.simulations[sim_id]

            # Check if footprint actually has any data during the inversion time range
            # (Particles may exit the domain before the inversion time range)
            if not sim.footprint.time_range.start < t_stop and sim.footprint.time_range.stop > t_start:
                continue

            print(f'Computing Jacobian row for {sim.id}...')

            # First spatially clip the footprints to slightly larger than the output grid
            footprint = sim.footprint.clip_to_grid(self.out_grid, buffer=self.grid_buffer)
            del sim.footprint  # free up memory

            # Sum the footprints over the inversion time bins
            foot = footprint.data.to_dataset()
            foot = integrate_over_time_bins(data=foot, time_bins=self.flux_times)\
                .rename({'time_bins': 'flux_time',
                        'run_time': 'obs_time'})

            # Regrid the footprints to the output grid using conservative regridding
            weights = self.regrid_weights.get('footprints')
            regridder = xe.Regridder(foot, self.out_grid, method='conservative',
                                     filename=weights)#, parallel=True if not weights else False)
            # Adding parallel=True is complicated - xesmf doesn't like how my datasets are chunked?
            #  When I do chunk the out_grid, I get:
            #    ValueError: zip() argument 2 is shorter than argument 1
            #  When I don't chunk the out_grid, I get:
            #    ValueError: Using `parallel=True` requires the output grid to have chunks along all spatial dimensions.
            #    If the dataset has no variables, consider adding an all-True spatial mask with appropriate chunks.

            if not weights:
                # Write weights to disk and save the filename to reuse
                self.regrid_weights['footprints'] = regridder.to_netcdf(filename=self._regrid_path / 'footprints.nc')

            regridded = regridder(foot, keep_attrs=True)

            # Stack the lon/lat dims into a single cell dim
            row = regridded.stack(cell=('lon', 'lat'))

            H_rows.append(row)

        H = xr.merge(H_rows).foot

        # Check for empty rows
        empty_rows = H.isnull().all(dim=['flux_time', 'cell'])
        if empty_rows.any():
            print('Warning: Empty rows in Jacobian matrix for receptors: '
                  f'{empty_rows.where(empty_rows, drop=True).stack(receptor_time=("receptor", "obs_time")).receptor_time.values}')
        else:
            print('Jacobian matrix built successfully.')

        # # Write jacobian to disk
        path.mkdir(exist_ok=True, parents=True)

        # Can't write python objects to disk
        # Need to serialize the IntervalIndex to a netcdf compatible format
        # Assign the midpoint to the flux_time coordinate
        # Create a new flux_time_bnds dim with the left and right bounds - size 2
        # Assign the left and right bounds to the flux_time_bnds coords
        
        # # Convert IntervalIndex to list of tuples
        # # How do i keep track of which side is closed?                                                                                                                                                                                                                                                                       
        # H = H.assign_coords(flux_t_start=('flux_time', [t.left for t in flux_times]),
        #                     flux_t_stop=('flux_time', [t.right for t in flux_times])).

        for flux_time in H.flux_t_start.values:
            print(f'Writing Jacobian matrix for {flux_time} to disk...')
            # Write each flux_time coordinate to a different file
            H.sel(flux_t_start=flux_time).to_netcdf(path / f'jacobian_{flux_time:%Y%m%d%H}.nc')

        return H

    def build_bio(self, bio: xr.DataArray) -> pd.Series:
        bio_fluxes = integrate_over_time_bins(data=bio, time_bins=self.flux_times)
        bio_concentrations = bio_fluxes * self.H  # TODO is this multiplied correctly?
        return bio_concentrations.as_pandas()
