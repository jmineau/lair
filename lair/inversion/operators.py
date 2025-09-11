import datetime as dt
from pathlib import Path
from typing import Literal, Self

import pandas as pd
import xarray as xr

from lair.inversion.core import ForwardOperator
from lair.inversion.flux import Jacobian
from lair.inversion.utils import integrate_over_time_bins
from lair.utils.geo import clip, write_rio_crs
from lair.utils.parallel import parallelize


class StiltJacobian(Jacobian):
    """
    STILT Jacobian

    This class implements a Jacobian matrix from STILT footprints.
    """

    from lair.air import stilt

    def __init__(self, data, **kwargs):
        super().__init__(data=data,
                         **kwargs)

    @classmethod
    def from_simulations(cls, simulations: list[Path],
                         flux_times: pd.IntervalIndex,
                         out_grid: xr.DataArray,
                         resolution: str | None = None,
                         subset_hours: int | list[int] | None = None,
                         num_processes: int | Literal['max'] = 1) -> Self:
        """
        Build a Jacobian matrix from a STILT model.

        Parameters
        ----------
        simulations : list[Path]
            List of paths to STILT simulation directories
        flux_times : pd.IntervalIndex
            Time bins for the fluxes
        out_grid : xr.DataArray
            Output grid for the fluxes. Must include CRS information.
        resolution : str | None, optional
            Resolution of the footprints to use, by default None (use highest resolution available)
        subset_hours : int | list[int] | None, optional
            Subset the simulations to specific hours of the day, by default None
        num_processes : int | Literal['max'], optional
            Number of processes to use for parallel computation, by default 1

        Returns
        -------
        StiltJacobian
            A STILT Jacobian matrix
        """

        print('Building Jacobian matrix...')

        # Assign rioxarray CRS to out_grid
        if not getattr(out_grid.rio, 'crs', None):
            crs = out_grid.attrs.get('crs', None)
            if crs is not None:
                out_grid = write_rio_crs(out_grid, crs)
            else:
                raise ValueError('out_grid must have CRS information in its attrs or be assigned with rioxarray')

        # Compute the Jacobian matrix in parallel
        H_rows = []
        missing_foots = []
        parallelized_computer = parallelize(cls._compute_jacobian_row, num_processes=num_processes)
        results = parallelized_computer(simulations,
                                        flux_times=flux_times, out_grid=out_grid,
                                        resolution=resolution, subset_hours=subset_hours)
        for row in results:
            if row is not None:
                if isinstance(row, xr.DataArray):
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

        print('Jacobian matrix built successfully.')

        jacobian = cls(H)
        jacobian.missing_foots = missing_foots
        return jacobian

    @staticmethod
    def _compute_jacobian_row(simulation: stilt.Simulation | Path,
                              flux_times: pd.IntervalIndex,
                              out_grid: xr.DataArray,
                              resolution: str | None = None,
                              subset_hours: int | list[int] | None = None
                              ) -> xr.DataArray | str | None:

        if isinstance(simulation, Path):
            sim = StiltJacobian.stilt.Simulation.from_path(simulation)
        elif isinstance(simulation, StiltJacobian.stilt.Simulation):
            sim = simulation
        else:
            raise ValueError('simulation must be a Path or a stilt.Simulation object')

        sim_id = sim.id

        if not sim.status == 'SUCCESS':
            return sim_id

        n_hours = sim.config.n_hours
        if n_hours >= 0:
            raise ValueError('STILT must be run backwards in time (n_hours < 0)')
        n_hours = pd.to_timedelta(n_hours, unit='h')

        t_start, t_stop = flux_times[0].left, flux_times[-1].right

        # Skip simulations that do not overlap with inversion time range
        sim_start = sim.receptor.time + n_hours
        sim_end = sim.receptor.time - pd.Timedelta(hours=1)
        if sim_end < t_start or sim_start >= t_stop:
            return None

        if subset_hours:
            # Subset simulations to specific hours
            if isinstance(subset_hours, int):
                subset_hours = [subset_hours]
        
            if not sim.receptor.time.dt.hour.isin(subset_hours):
                return None

        # Load footprint object
        if resolution:
            # Load footprint at specified resolution
            footprint = sim.footprints[resolution]
        else:
            # Get the default (highest) resolution footprint
            footprint = sim.footprint

        if not footprint:
            return None

        # Check if footprint time range overlaps with inversion time range
        if not (footprint.time_range[0] < t_stop and footprint.time_range[1] > t_start):
            # A footprint within a simulation might not overlap with the inversion time range
            # even if the simulation does
            return None

        print(f'Computing Jacobian row for {sim_id}...')

        # Clip the footprint to the output grid
        foot = StiltJacobian._clip_footprint_to_grid(footprint=footprint, grid=out_grid)

        if foot.size == 0:
            return None

        # Integrate each simulation over flux_time bins
        foot = integrate_over_time_bins(data=foot, time_bins=flux_times)
        foot = foot.rename({'time_bins': 'flux_time'})

        # Expand dimensions to include obs_time & obs_location
        foot = foot.expand_dims({'obs_time': [sim.receptor.time],
                                 'obs_location': [sim.receptor.location.id]})

        return foot

    @staticmethod
    def _clip_footprint_to_grid(footprint: stilt.Footprint, grid: xr.DataArray
                           ) -> xr.DataArray:
        """
        Clip the footprint to the output grid + buffer

        Parameters
        ----------
        footprint : stilt.Footprint
            Footprint object
        grid : xr.DataArray
            Output grid
        buffer : float
            Buffer around the output grid to clip the footprints

        Returns
        -------
        xr.DataArray
            Clipped footprint
        """
        foot = footprint.data

        # Assign CRS to footprint if not already assigned
        if not getattr(foot.rio, 'crs', None):
            foot = write_rio_crs(foot, crs=footprint.projection)

        # Clip the footprint to the output grid bounds
        bbox = grid.rio.bounds()

        clipped = clip(foot, bbox=bbox, crs=grid.rio.crs)
        return clipped
