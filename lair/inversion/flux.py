"""
Bayesian Flux Inversion
"""

from abc import ABC
import datetime as dt
from functools import cached_property
from pathlib import Path
import shutil
from typing import Any, Literal
from typing_extensions import \
    Self  # requires python 3.11 to import from typing

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from lair.air.stilt import Model as STILTModel
from lair.air.stilt import Receptor, Footprint
from lair import inventories
from lair.inventories import Inventory
from lair.inversion.bayesian import Inversion
from lair.utils.clock import regular_times_to_intervals, time_decay_matrix
from lair.utils.geo import earth_radius
from lair.utils.parallel import parallelize
from sklearn.metrics.pairwise import haversine_distances


# TODO:
# - non lat-lon grids
#   - polygonal flux "cells"
# - multiple receptors

# FIXME
# - I don't know what the correct way to regrid footprints is / if there is a good way
# - Perhaps I should just set the bounds and force everything to be on the same grid as the footprint?


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


def generate_flux_index_from_latlon_grid(t_bins: pd.IntervalIndex,
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
        MultiIndex for flux state: [flux_time, lat, lon]
    """

    x_dim, y_dim = 'lon', 'lat'
    x, y = out_grid[x_dim].values, out_grid[y_dim].values
    xx, yy = np.meshgrid(x, y)
    y_x_tuples = pd.Index(zip(yy.ravel(), xx.ravel())).values.reshape(xx.shape)
    cells = xr.DataArray(y_x_tuples, coords={y_dim: y, x_dim: x})

    if grid_mask is not None:
        # Mask grid cells where mask == 0
        cells = cells.where(grid_mask)

    flux_index = pd.MultiIndex.from_product([t_bins, y, x],
                                            names=['flux_time', 'lat', 'lon'])
    return flux_index


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


def stack_obs_dims(data: xr.DataArray, obs_dims: list[str]) -> xr.DataArray:
    """
    Stack the observation dimensions into a single 'obs' dimension.

    Parameters
    ----------
    data : xr.DataArray
        Data to stack.
    obs_dims : list[str]
        Observation dimensions to stack.

    Returns
    -------
    xr.DataArray
        Stacked observations.
    """
    return data.stack(obs=obs_dims)


def stack_flux_dims(data: xr.DataArray, flux_dims: list[str]) -> xr.DataArray:
    """
    Stack the flux dimensions into a single 'flux' dimension.

    Parameters
    ----------
    data : xr.DataArray
        Data to stack.
    flux_dims : list[str]
        Flux dimensions to stack.

    Returns
    -------
    xr.DataArray
        Stacked state.
    """
    return data.stack(flux=flux_dims)


class CSVMixin(ABC):
    obs_dims: list[str]
    data: pd.Series

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        """
        Read the data from a CSV file.

        Parameters
        ----------
        path : str | Path
            Path to the CSV file.

        Returns
        -------
        CSVMixin
            Instance of CSVMixin.
        """
        data = pd.read_csv(path, index_col=cls.obs_dims)
        return cls(data)  # TODO: do i need to convert this to a series?

    def to_csv(self, path: str | Path) -> None:
        """
        Write the data to a CSV file.

        Parameters
        ----------
        path : str | Path
            Path to the CSV file.
        """
        self.data.to_csv(path)


class ObsMixin(ABC):
    obs_dims: list[str] = ['receptor', 'obs_time']  # TODO maybe in the future this will be set by init

    def __init__(self, data: pd.Series):
        self._check_obs_dims(data)
        data = data.rename(self.__class__.__name__.lower())  #  rename for consistency
        self.data = data

    def _check_obs_dims(self, data: pd.Series) -> None:
        # Can be single index with just time or multiindex with receptor and time
        if isinstance(data.index, pd.MultiIndex):
            assert data.index.names == self.obs_dims, f'index must have names: {self.obs_dims}'
        else:
            assert data.index.name == self.obs_dims[-1], f'index must have name: {self.obs_dims[-1]}'

    @classmethod
    def stack_obs_dims(cls, data: xr.DataArray) -> xr.DataArray:
        """
        Stack the observation dimensions into a single 'obs' dimension.

        Parameters
        ----------
        data : xr.DataArray
            Data to stack.

        Returns
        -------
        xr.DataArray
            Stacked observations.
        """
        obs_dims = [dim for dim in cls.obs_dims if dim in data.dims]
        return stack_obs_dims(data, obs_dims=obs_dims)


class Observation(ObsMixin, CSVMixin):

    @property
    def z(self) -> xr.DataArray:
        return self.stack_obs_dims(self.data.to_xarray())


class Background(ObsMixin, CSVMixin):

    @property
    def c(self) -> xr.DataArray:
        return self.stack_obs_dims(self.data.to_xarray())

    def __add__(self, other: Self) -> Self:
        """
        Add two Background instances.

        Parameters
        ----------
        other : Background
            Another Background instance.

        Returns
        -------
        Background
            Sum of the two Background instances.
        """
        assert isinstance(other, Background), 'Can only add Background instances'
        return self.__class__(self.data + other.data)

    @classmethod
    def from_flux(cls, flux: xr.DataArray, jacobian: 'Jacobian') -> Self:
        """
        Generate a constant background from a flux field.

        Parameters
        ----------
        flux : xr.DataArray
            Flux field. Must have one fewer dimension than the Jacobian matrix.
        jacobian : Jacobian
            Jacobian matrix.

        Returns
        -------
        Background
            Background data.
        """
        # Multiply the flux field by the Jacobian matrix
        c = flux @ jacobian.data
        c = c.to_series()  # convert to pandas series

        # Create dynamic class based on the name of the flux array subclassing Background
        name = ''.join([i.capitalize() for i in flux.name.split('_')])
        cls_name = f'{name}Background'
        cls = type(cls_name, (Background,), {})
        return cls(c)


class FluxMixin(ABC):

    data: xr.DataArray
    flux_dims: list[str] = ['flux_time', 'lat', 'lon']  # TODO maybe in the future this will be set by init

    def __init__(self, data: xr.DataArray):
        self._check_flux_dims(data)
        self.data = data

        # Store regridder if it was used
        self._regridder = None

    def _check_flux_dims(self, data: xr.DataArray) -> None:
        assert all(dim in data.dims for dim in self.flux_dims), f'data must have dimensions: {self.flux_dims}'

    @classmethod
    def stack_flux_dims(cls, data: xr.DataArray) -> xr.DataArray:
        """
        Stack the flux dimensions into a single 'flux' dimension.

        Parameters
        ----------
        data : xr.DataArray
            Data to stack.

        Returns
        -------
        xr.DataArray
            Stacked state.
        """
        return stack_flux_dims(data, flux_dims=cls.flux_dims)

    @staticmethod
    def encode_flux_times(data: xr.Dataset) -> xr.Dataset:
        # Assign the start time as the flux_time coordinate
        # and add an additional flux_t_stop coord to the flux_time dimension
        flux_times = data.get_index('flux_time')
        data = data.assign_coords(
            flux_time=flux_times.left,
            flux_t_stop=("flux_time", flux_times.right)
        )

        # Add attributes
        data.attrs['flux_time_closed'] = flux_times.closed
        return data

    @staticmethod
    def decode_flux_times(data: xr.Dataset) -> xr.Dataset:
        # Rebuild the IntervalIndex with the correct closed attribute
        flux_times = pd.IntervalIndex.from_arrays(left=data.flux_time.values,
                                                  right=data.flux_t_stop.values,
                                                  closed=data.attrs.pop('flux_time_closed'))
        data = data.drop('flux_t_stop')  # drop the flux_t_stop coord

        # Assign the rebuilt IntervalIndex to the data
        data = data.assign_coords(flux_time=flux_times)
        return data

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        """
        Read the Jacobian matrix from disk.

        Parameters
        ----------
        path : str | Path
            Path to the Jacobian matrix. If a directory is given, all flux time slices in the directory are read.
            If a single file is given, only that flux time slice is read.

        Returns
        -------
        Jacobian
            Jacobian matrix.
        """
        path = Path(path)
        if path.is_dir():
            # Read all netcdf files in the directory
            data = xr.open_mfdataset(str(path / '*.nc'))
        else:
            # Read the single file
            data = xr.open_dataset(path)

        # Rebuild the IntervalIndex with the correct closed attribute
        data = cls.decode_flux_times(data)

        var = list(data.data_vars)[0]  # should only be one variable
        return cls(data[var])

    def to_netcdf(self, path: str | Path) -> None:
        """
        Write the state matrix to disk.

        Parameters
        ----------
        path : str | Path
            Path to write the state matrix.
        groups : str, optional
            Group the data by a variable before writing to disk, by default None.
        """
        path = Path(path)
        if path.suffix == '':  # check if path is a directory
            # use path.suffix instead of path.is_dir() because is_dir() doesn't work with relative paths
            multifile = True
            path.mkdir(exist_ok=True, parents=True)
        else:
            if not path.parent.exists():
                path.parent.mkdir(parents=True)
            multifile = False

        data = self.data.to_dataset()  # convert to dataset to write to disk

        # Can't write python objects to disk
        # Need to serialize the IntervalIndex to a netcdf compatible format
        data = self.encode_flux_times(data)

        # Write the data to disk
        suffix = self.__class__.__name__.lower()
        if multifile:
            slice_times, flux_slices = zip(*data.groupby('flux_time'))
            flux_paths = [path / f'{pd.Timestamp(flux_time):%Y%m%d%H}_{suffix}.nc'
                        for flux_time in slice_times]
            xr.save_mfdataset(flux_slices, paths=flux_paths)
        else:
            data.to_netcdf(path)


class Prior(FluxMixin):

    @property
    def x_0(self) -> xr.DataArray:
        return self.stack_flux_dims(self.data)

    @classmethod
    def from_inventory(cls, inventory: Inventory, flux_times, out_grid,
                       units: str | None = None,
                       regrid_weights: str | Path | None = None) -> Self:
        pollutant = inventory.pollutant
        time_step = inventory.time_step
        crs = inventory.crs

        print('Building prior state matrix...')
        # Sum sectors
        total = inventories.sum_sectors(inventory.data)

        # Wrap in Inventory object
        wrapped = inventories.Inventory(total.to_dataset(), pollutant=pollutant, time_step=time_step,
                                        src_units=total.attrs['units'], crs=crs)

        # Convert units
        if units:
            wrapped = wrapped.convert_units(units)
        emis = wrapped.data['emissions']

        # Regrid to out_grid
        regridder = xe.Regridder(emis, out_grid, method='conservative',
                                 weights=regrid_weights)
        prior = regridder(emis)

        # Convert time coords to IntervalIndex
        intervalindex = regular_times_to_intervals(prior.time, time_step=time_step,
                                                   closed='left')
        prior = prior.assign_coords(time=intervalindex)
        prior = prior.rename({'time': 'flux_time'})

        # Filter to flux_times
        prior = prior.sel(flux_time=flux_times)

        prior.name = 'prior_flux'

        print('Prior state matrix built successfully.')

        prior = cls(prior)
        prior._regridder = regridder  # store regridder for later use
        return prior


class Jacobian(ObsMixin, FluxMixin):
    MISSING_FOOTPRINT = object()

    def __init__(self, data: xr.DataArray):
        FluxMixin.__init__(self, data)

    @property
    def H(self) -> xr.DataArray:
        data = self.stack_obs_dims(self.data)
        data = self.stack_flux_dims(data)
        return data

    @staticmethod
    def _compute_jacobian_row_from_stilt(sim_id, stilt, t_start, t_stop, flux_times,
                                         out_grid, grid_buffer, regrid_weights) -> xr.DataArray | None:
        sim = stilt.simulations[sim_id]

        if not sim.has_footprint:
            return sim_id

        if not sim.footprint.time_range.start < t_stop and sim.footprint.time_range.stop > t_start:
            return None

        print(f'Computing Jacobian row for {sim.id}...')

        footprint = sim.footprint.clip_to_grid(out_grid, buffer=grid_buffer)
        del sim.footprint  # free up memory

        # Integrate each simulation over flux_time bins
        foot = footprint.data.to_dataset()
        foot = integrate_over_time_bins(data=foot, time_bins=flux_times)
        foot = foot.rename({'time_bins': 'flux_time', 'run_time': 'obs_time'})

        # FIXME THIS IS NOT HOW FOOTPRINTS SHOULD BE REGRIDDED
        # Regrid the footprint to the output grid
        # regridder = xe.Regridder(ds_in=foot, ds_out=out_grid, method='conservative',
        #                          weights=regrid_weights)
        # foot = regridder(foot, keep_attrs=True)
        return foot

    @classmethod
    def from_path(cls, path):
        jacobian = super().from_path(path)

        # Nans can be introduced when combined jacobian files that were not computed together
        # fill with 0  TODO: can I set the fillvalue to automatically do this?
        jacobian.data = jacobian.data.fillna(0)
        return jacobian

    @classmethod
    def from_stilt(cls, stilt: STILTModel, flux_times: pd.IntervalIndex,
                   out_grid: xr.DataArray, grid_buffer: float = 0.1,
                   subset_hours: int | list[int] | None = None,
                   regrid_weights: str | Path | None = None,
                   num_processes: int | Literal['max'] = 1) -> Self:

        print('Building Jacobian matrix...')

        # Get start and end times for the inversion
        t_start, t_stop = flux_times[0].left, flux_times[-1].right

        # Get successful simulations within the inversion time range
        print('Getting simulations within inversion time range...')
        assert stilt.n_hours < 0, 'STILT must be ran backwards in time'
        sim_df = stilt.simulations.to_dataframe(subset='successful').reset_index()
        assert len(sim_df) > 0, 'No successful simulations found'
        sim_df['sim_t_start'] = sim_df.run_time + pd.Timedelta(hours=stilt.n_hours)
        sim_df['sim_t_end'] = sim_df.run_time - pd.Timedelta(hours=1)
        sim_df = sim_df[(sim_df.sim_t_start < t_stop) & (sim_df.sim_t_end >= t_start)]

        # Subset simulations to specific hours
        if subset_hours:
            if isinstance(subset_hours, int):
                subset_hours = [subset_hours]
            sim_df = sim_df[sim_df.run_time.dt.hour.isin(subset_hours)]

        # Compute the xesmf weights for regridding the footprints
        # if regrid_weights is not None and not Path(regrid_weights).exists():
        #     print('Computing regrid weights...')
        #     sim = stilt.simulations[sim_df['sim_id'].iloc[0]]
        #     footprint = sim.footprint.clip_to_grid(out_grid, buffer=grid_buffer)
        #     foot = footprint.data.to_dataset()
        #     regridder = xe.Regridder(ds_in=foot, ds_out=out_grid, method='conservative')
        #     regridder.to_netcdf(filename=regrid_weights)
        # else:
        regridder = None

        # Compute the Jacobian matrix in parallel
        H_rows = []
        missing_foots = []
        parallelized_computer = parallelize(cls._compute_jacobian_row_from_stilt, num_processes=num_processes)
        results = parallelized_computer(sim_df['sim_id'],
                                        stilt=stilt, t_start=t_start, t_stop=t_stop,
                                        flux_times=flux_times, out_grid=out_grid,
                                        grid_buffer=grid_buffer, regrid_weights=regrid_weights)
        for row in results:
            if row is not None:
                if isinstance(row, xr.Dataset):
                    H_rows.append(row)
                elif isinstance(row, str):
                    missing_foots.append(row)
                else:
                    raise ValueError('Unexpected output from compute_jacobian_row')

        H = xr.merge(H_rows).foot
        H.name = 'jacobian'

        # Reset attrs
        del H.attrs['standard_name']
        del H.attrs['long_name']
        H.flux_time.attrs = {}  # drop flux_time attrs from stilt

        empty_rows = H.isnull().all(dim=cls.flux_dims)  # TODO is there ever going to be empty rows?
        if empty_rows.any():
            print('Warning: Empty rows in Jacobian matrix for receptors: '
                  f'{empty_rows.where(empty_rows, drop=True).stack(receptor_time=("receptor", "obs_time")).receptor_time.values}')
        else:
            print('Jacobian matrix built successfully.')

        jacobian = cls(H)
        jacobian._regridder = regridder
        jacobian.missing_foots = missing_foots
        return jacobian


class CovarianceMixin(ABC):

    dim: str
    data: np.ndarray

    def __init__(self, data: np.ndarray, index):
        self.data = data
        self.index = index

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        """
        Read the covariance matrix from disk.

        Parameters
        ----------
        path : str | Path
            Path to the covariance matrix.

        Returns
        -------
        CovarianceMixin
            Covariance matrix.
        """
        return cls(np.load(path))  # FIXME need to pass index somehow

    def to_npy(self, path: str | Path) -> None:
        """
        Write the covariance matrix to disk.

        Parameters
        ----------
        path : str | Path
            Path to write the covariance matrix.
        """
        np.save(path, self.data)

    def loc(self, index: pd.Index) -> Self:
        """
        Get the covariance matrix values for a given index.

        Parameters
        ----------
        index : pd.Index
            Index to get covariance matrix values for.

        Returns
        -------
        CovarianceMixin
            Covariance matrix values for the given index
        """
        df = pd.DataFrame(self.data, index=self.index, columns=self.index)
        return self.__class__(df.loc[index, index].values, index=index)

    def sort(self, index: pd.Index) -> Self:
        """
        Sort the covariance matrix by index.

        Parameters
        ----------
        index : pd.Index
            Index to sort by.

        Returns
        -------
        CovarianceMixin
            Sorted covariance matrix.
        """
        return self.__class__(self.loc(index), index=index)


class PriorErrorCovariance(CovarianceMixin):
    """
    Prior Error Covariance matrix for the inversion.
    Has dimensions of flux_index x flux_index.
    """
    dim: str = 'state'

    @classmethod
    def from_variances(cls, prior_error_variances: float | np.ndarray,
                       flux_index: pd.MultiIndex,
                       spatial_corr: dict | np.ndarray | None = None,
                       temporal_corr: dict | np.ndarray | None = None) -> Self:
        """
        Create a PriorCovariance instance from variances and correlation matrices.

        Parameters
        ----------
        prior_error_variances : float | np.ndarray
            Prior error variances for the inversion.
        flux_index : pd.MultiIndex
            MultiIndex for flux state.
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
        PriorCovariance
            Instance of PriorCovariance.
        """
        flux_times = flux_index.get_level_values('flux_time').unique()
        lats = flux_index.get_level_values('lat').unique()
        lons = flux_index.get_level_values('lon').unique()

        if spatial_corr is None:
            spatial_corr = np.eye(len(lats) * len(lons))
        elif isinstance(spatial_corr, dict):
            spatial_corr = cls.build_spatial_corr(lat=lats, lon=lons, **spatial_corr)
        elif not isinstance(spatial_corr, np.ndarray):
            raise ValueError('spatial_corr must be a dict or np.ndarray')

        if temporal_corr is None:
            temporal_corr = np.eye(len(flux_times))
        elif isinstance(temporal_corr, dict):
            temporal_corr = cls.build_temporal_corr(times=flux_times.left, **temporal_corr)
        elif not isinstance(temporal_corr, np.ndarray):
            raise ValueError('temporal_corr must be a dict or np.ndarray')

        S_0 = (prior_error_variances ** 2) * np.kron(temporal_corr, spatial_corr)
        return cls(S_0, index=flux_index)

    @staticmethod
    def build_spatial_corr(lat: np.ndarray, lon: np.ndarray, method: str | dict | None = None, **kwargs) -> np.ndarray:
        """
        Build the spatial error correlation matrix for the inversion.
        Has dimensions of lat x lon.

        Parameters
        ----------
        lat : np.ndarray
            Latitude coordinates of the flux cells.
        lon : np.ndarray
            Longitude coordinates of the flux cells.
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
        np.ndarray
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
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        cells = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])  # create pairwise stacks of lat and lon
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

                coarse_grid = kwargs['coarse_grid']
                regridder = kwargs['regridder']

                # Reshape the weights to the shape of the regridder and assign coords
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
                        method_corr[np.ix_(shared_out_cells, shared_out_cells)] = np.outer(values, values)

                # Overwrite the diagonal elements with 1
                np.fill_diagonal(method_corr, 1)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            # Combine the correlation matrices based on the method weights
            corr += weight * method_corr

        return corr

    @staticmethod
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
        np.ndarray
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
                method_corr = time_decay_matrix(times, decay=kwargs['time_scale'])

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


class ModelDataMismatch(CovarianceMixin):
    """
    Model-data mismatch matrix for the inversion.
    Has dimensions of obs_index x obs_index.
    """
    dim: str = 'obs'

    @classmethod
    def from_rmse(cls, rmse: float, obs_index: pd.MultiIndex,
                  time_decay: str | pd.Timedelta | bool | None = None) -> Self:
        """
        Create a MDM component from the root mean squared error (RMSE).

        Parameters
        ----------
        rmse : float
            Root mean squared error.
        obs_index : pd.MultiIndex
            MultiIndex for observations.
        time_decay : str | pd.Timedelta | bool, optional
            Time decay for the MDM component, by default None.
            If None, the MDM component has diagonal elements set to RMSE^2.
            If False, all elements are set to RMSE^2.
            If provided as a time scale, the MDM component is a diagonal matrix
            with diagonal elements set to RMSE^2
            and off-diagonal elements set to RMSE^2 * exp(-|t_i - t_j| / time_decay

        Returns
        -------
        ModelDataMismatch
            MDM component.
        """
        # TODO:
        # - accept array of RMSEs
        # - spatial decay based on receptor locations

        mdm = np.eye(len(obs_index)) * rmse ** 2
        off_diag_mask = ~np.eye(len(obs_index), dtype=bool)

        if time_decay:
            # Compute off-diagonal elements as exponential time decay
            off_diags = np.zeros((len(obs_index), len(obs_index)))

            # Calculate the time differences for each receptor
            for receptor, group in obs_index.to_frame(index=False).groupby('receptor'):
                receptor_indices = group.index
                receptor_times = group['obs_time']
                decay_matrix = time_decay_matrix(receptor_times, decay=time_decay)
                off_diags[np.ix_(receptor_indices, receptor_indices)] = decay_matrix

            mdm[off_diag_mask] += off_diags[off_diag_mask] * (rmse ** 2)
        elif time_decay is False:
            # Set all elements to the RMSE^2
            mdm[:] = rmse ** 2
        # Else leave as identity matrix

        return cls(mdm, index=obs_index)

    def __add__(self, other: Self) -> Self:
        """
        Add two ModelDataMismatch instances.

        Parameters
        ----------
        other : ModelDataMismatch
            Another ModelDataMismatch instance.

        Returns
        -------
        ModelDataMismatch
            Sum of the two ModelDataMismatch instances.
        """
        assert isinstance(other, ModelDataMismatch), 'Can only add ModelDataMismatch instances'
        assert self.index.equals(other.index), 'ModelDataMismatch instances must have the same index'
        assert self.data.shape == other.data.shape, 'ModelDataMismatch instances must have the same shape'
        return ModelDataMismatch(self.data + other.data, index=self.index)


class FluxInversion(Inversion):
    """
    Bayesian flux inversion
    """

    _inputs = ["obs", "background", "bio", "prior", 'jacobian',
               "prior_error_cov", "modeldata_mismatch"]

    serializers = {
        'obs': {
            'save': lambda obj, path: obj.to_csv(path),
            'load': lambda path: Observation.from_path(path)
        },
        'background': {
            'save': lambda obj, path: obj.to_csv(path),
            'load': lambda path: Background.from_path(path)
        },
        'prior': {
            'save': lambda obj, path: obj.to_netcdf(path),
            'load': lambda path: Prior.from_path(path)
        },
        'jacobian': {
            'save': lambda obj, path: obj.to_netcdf(path),
            'load': lambda path: Jacobian.from_path(path)
        },
        'prior_error_cov': {
            'save': lambda obj, path: obj.to_npy(path),
            'load': lambda path: PriorErrorCovariance.from_path(path)
        },
        'modeldata_mismatch': {
            'save': lambda obj, path: obj.to_npy(path),
            'load': lambda path: ModelDataMismatch.from_path(path)
        }
    }

    # Default paths as class attribute
    default_paths = {
        'background': 'background.csv',
        'jacobian': 'jacobian',
        'modeldata_mismatch': 'modeldata_mismatch.npy',
        'obs': 'obs.csv',
        'posterior': 'posterior.nc',
        'prior': 'prior.nc',
        'prior_error_cov': 'prior_error_cov.npy',
    }

    def __init__(self,
                 project: str | Path,
                 flux_times: pd.IntervalIndex,
                 out_grid: xr.DataArray,
                 obs: Observation | None = None,
                 background: Background | None = None,
                 bio: pd.Series | xr.DataArray | None = None,
                 prior: Prior | None = None,
                 jacobian: Jacobian | None = None,
                 prior_error_cov: PriorErrorCovariance | None = None,
                 modeldata_mismatch: ModelDataMismatch | None = None,
                 grid_mask: xr.DataArray | None = None,
                 paths: dict[str, str | Path] | None = None,
                 ) -> None:
        # TODO should probably allow setting obs_times/index from here?
        # Set project directory
        self.path = Path(project)
        self.project = self.path.name
        self.path.mkdir(exist_ok=True, parents=True)  # create project directory if it doesn't exist

        # Set paths
        if paths:
            # Merge default paths with user-specified paths
            self.paths = {key: self.path / paths.get(key, default)
                          for key, default in self.default_paths.items()}
        else:
            # Use default paths
            self.paths = {key: self.path / default
                          for key, default in self.default_paths.items()}

        # Define state
        self.flux_times = flux_times
        self.out_grid = out_grid
        self.grid_mask = grid_mask  # TODO implment masking

        # Load or build inputs 
        self.obs = obs if obs is not None else self._load_input('obs')
        self.background = background if background is not None else self._load_input('background')
        self.bio = bio if not self.paths.get('bio') else self._load_input('bio')
        self.prior = prior if prior is not None else self._load_input('prior')
        self.jacobian = jacobian if jacobian is not None else self._load_input('jacobian')
        self.prior_error_cov = prior_error_cov if prior_error_cov is not None else self._load_input('prior_error_cov')
        self.modeldata_mismatch = modeldata_mismatch if modeldata_mismatch is not None else self._load_input('modeldata_mismatch')
        
        # Align indices
        self.obs_index = None
        self.flux_index = None
        self._align_indices()

        # Save aligned inputs
        self._save_inputs()

        # Get coordinates
        self.receptors = self.obs_index.get_level_values('receptor').unique()
        self.obs_times = self.obs_index.get_level_values('obs_time').unique()
        # self.flux_times = self.flux_index.get_level_values('flux_time').unique()
        self.lats = self.flux_index.get_level_values('lat').unique()
        self.lons = self.flux_index.get_level_values('lon').unique()

        # Initialize inversion
        super().__init__(z=self.obs.z.values, c=self.background.c.values,
                         x_0=self.prior.x_0.values, H=self.jacobian.H.values,
                         S_0=self.prior_error_cov.data, S_z=self.modeldata_mismatch.data)

    def run(self):
        """
        Run the inversion.
        """
        print('Running inversion...')
        # save outputs
        print('Inversion complete.')

    @cached_property
    def posterior(self) -> xr.DataArray:
        """
        Posterior flux estimate.
        """
        x_hat = super().x_hat
        da = xr.DataArray(x_hat, coords={'flux': self.flux_index}, dims=['flux'])
        return da.unstack('flux')

    @cached_property
    def posterior_obs(self) -> pd.Series:
        """
        Posterior observation estimate.
        """
        y_hat = super().y_hat
        return pd.Series(y_hat, index=self.obs_index)

    @cached_property
    def posterior_error_cov(self) -> np.ndarray:
        """
        Posterior error covariance matrix.
        """
        S_hat = super().S_hat
        return S_hat
    
    def _align_indices(self) -> None:
        print('Aligning indices...')
        # Align inputs
        inputs = [self.obs.z, self.background.c, self.prior.x_0, self.jacobian.H]
        aligned_inputs = xr.align(*inputs, join='inner')

        self.obs.data = aligned_inputs[0].to_pandas()
        self.background.data = aligned_inputs[1].to_pandas()
        self.prior.data = aligned_inputs[2].unstack()
        self.jacobian.data = aligned_inputs[3].unstack()

        # Get new indices
        self.obs_index = self.obs.data.index
        self.flux_index = aligned_inputs[3].get_index('flux')

        # Align covariance matrices
        self.prior_error_cov = self.prior_error_cov.loc(self.flux_index)
        self.modeldata_mismatch = self.modeldata_mismatch.loc(self.obs_index)

    def _save_inputs(self):
        """Saves all provided inputs to individual files in the project directory."""
        print('Saving inputs...')
        for input_name in self._inputs:
            print(f'Saving {input_name}')
            obj = getattr(self, input_name)
            if obj is not None:
                path = self.paths[input_name]
                if path.exists():
                    # Remove existing file or directory
                    if path.suffix == '':
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink()
                self.serializers[input_name]['save'](obj, path)

    def _load_input(self, input_name: str):
        """Loads an input from disk."""
        print(f'Loading {input_name}')
        path = self.paths[input_name]
        return self.serializers[input_name]['load'](path)
