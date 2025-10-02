"""
Flux inversion module.

This module provides classes and utilities for performing atmospheric flux inversion,
a technique used to estimate surface fluxes (such as greenhouse gas emissions or uptake)
from observed atmospheric concentrations. Flux inversion is a specific application of
the general inverse problem framework, where the goal is to infer unknown fluxes
(posterior) given observed concentrations, a prior flux inventory, and a model
of atmospheric transport.

Key terminology differences from the base inverse problem:
- "Prior" refers to the initial estimate of surface fluxes (e.g., from inventories or models).
- "Posterior" refers to the updated estimate of fluxes after assimilating observations.
- "Jacobian" (or "Forward Operator") maps fluxes to concentrations via atmospheric transport.
- "Concentrations" are the observed values at receptor locations/times.
- "Background" is the baseline (constant) concentration not attributed to local fluxes.

The Jacobian matrix is constructed using atmospheric transport models, currently STILT
(Stochastic Time-Inverted Lagrangian Transport), which simulates the influence of surface
fluxes on observed concentrations by generating footprints for each observation.
These footprints quantify the sensitivity of each observation to fluxes at different
locations and times, forming the basis of the Jacobian.
This module supports building the Jacobian from STILT simulations, specifying time bins,
grid resolutions, and parallel computation. It also provides the FluxInversion class,
which extends the base InverseProblem to handle flux-specific terminology and plotting
interfaces for visualizing results.
"""

from collections import defaultdict
import datetime as dt
from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lair.air import stilt
from lair.utils.parallel import parallelize
from lair.utils.geo import PC

from lair.inversion import estimators  # import to register estimators
from lair.inversion.core import (
    SymmetricMatrix,
    Estimator,
    ForwardOperator as Jacobian,
    InverseProblem
)
from lair.inversion.utils import integrate_over_time_bins

# TODO:
    # eventually want to support multiple flux source (farfield/bio/etc)
    # enable regridding
    # build stilt jacobian from geometries or nested grid
    # ability to extend state elements


class StiltJacobianBuilder:
    """
    Build the Jacobian matrix from STILT simulations.

    Attributes
    ----------
    simulations : list[stilt.Simulation | Path]
        List of STILT simulations or paths to simulation directories.
    failed_sims : list[str]
        List of simulation IDs that failed to process.

    Methods
    -------
    build_from_coords(coords, flux_times, resolution=None, subset_hours=None, num_processes=1)
        Build the Jacobian matrix H from specified coordinates (x, y) and flux time bins.
    """

    def __init__(self, simulations: List[stilt.Simulation | Path]):
        """
        Initialize the Jacobian builder with a list of STILT simulations
        or paths to simulation directories.

        Parameters
        ----------
        simulations : list[stilt.Simulation | Path]
            List of STILT simulations or paths to simulation directories.
        """
        self.simulations = simulations

        self.failed_sims = []

    def build_from_coords(self,
                          coords: list[tuple[float, float]] | dict[str, list[tuple[float, float]]],
                          flux_times: pd.IntervalIndex,
                          resolution: str | None = None,
                          subset_hours: int | list[int] | None = None,
                          num_processes: int | Literal['max'] = 1,
                          location_mapper: dict[str, str] | None = None,
                          ) -> Jacobian | dict[str, Jacobian]:
        """
        Build the Jacobian matrix H from specified coordinates (x, y) and flux time bins.

        Parameters
        ----------
        coords : list[tuple[float, float]] | dict[str, list[tuple[float, float]]]
            Coordinates of the output grid points.
            Multiple sets of coordinates can be provided as a dictionary of lists of coordinate tuples.
            Otherwise, a single list of coordinate tuples can be provided.
            Coordinates should be in the same CRS as the STILT footprints and specified as (x, y) tuples.
        flux_times : pd.IntervalIndex
            Time bins for the fluxes
        resolution : str | None, optional
            Resolution of the footprints to use, by default None (use highest resolution available)
        subset_hours : int | list[int] | None, optional
            Subset the simulations to specific hours of the day, by default None
        num_processes : int | Literal['max'], optional
            Number of processes to use for parallel computation, by default 1
        location_mapper : dict[str, str] | None, optional
            Optional mapping of observation location IDs to new IDs.

        Returns
        -------
        Jacobian | dict[str, Jacobian]
            If coords is a dict, returns a dict of Jacobians for each set of coordinates.
            Otherwise, returns a single Jacobian.
        """

        print('Building Jacobian matrix...')

        if not isinstance(coords, dict):
            coords = {'DEFAULT': coords}

        # Build the Jacobian matrix in parallel
        H_rows = defaultdict(list)
        parallelized_builder = parallelize(self._build_jacobian_row_from_coords,
                                           num_processes=num_processes)
        results = parallelized_builder(self.simulations,
                                       coords=coords,
                                       flux_times=flux_times,
                                       resolution=resolution,
                                       subset_hours=subset_hours)
        for row in results:
            if row is not None:
                if isinstance(row, dict):
                    for key, df in row.items():
                        H_rows[key].append(df)
                elif isinstance(row, str):
                    if row not in self.failed_sims:
                        self.failed_sims.append(row)
                else:
                    raise ValueError('Unexpected output from build_jacobian_row')

        H_dict = {}
        for key, rows in H_rows.items():
            if rows:
                print(f'Combining {len(rows)} rows for {key} jacobian...')
                H = pd.concat(rows).fillna(0)

                if location_mapper:
                    h = H.reset_index()
                    h['obs_location'] = h['obs_location'].map(location_mapper).fillna(h['obs_location'])
                    H = h.set_index(['obs_location', 'obs_time'])

                H = Jacobian(H)
                H_dict[key] = H

                if key == 'DEFAULT':
                    return H

        print('Jacobian matrix built successfully.')

        return H_dict

    @staticmethod
    def _build_jacobian_row_from_coords(simulation: stilt.Simulation | Path,
                                        coords: dict[str, list[tuple[float, float]]],
                                        flux_times: pd.IntervalIndex,
                                        resolution: str | None = None,
                                        subset_hours: int | list[int] | None = None
                                        ) -> dict[str, pd.DataFrame] | str | None:
        """
        Build a row of the Jacobian matrix for a single STILT simulation
        """
        t_start, t_stop = flux_times[0].left, flux_times[-1].right
        
        # Get simulation object
        sim = StiltJacobianBuilder._get_sim(simulation=simulation,
                                            t_start=t_start, t_stop=t_stop,
                                            subset_hours=subset_hours)
        if not isinstance(sim, stilt.Simulation):
            return sim  # could be None or sim.id if failed

        # Get footprint for the simulation
        footprint = StiltJacobianBuilder._get_footprint(sim=sim,
                                                        t_start=t_start, t_stop=t_stop,
                                                        resolution=resolution)
        if footprint is None:
            return None

        print(f'Computing Jacobian row for {sim.id}...')

        # Convert xarray to pandas for sparse representation
        foot = footprint.data.to_series()

        # Get the x and y dimension names
        is_latlon = 'lon' in foot.index.names and 'lat' in foot.index.names
        x_dim = 'lon' if is_latlon else 'x'
        y_dim = 'lat' if is_latlon else 'y'

        foot = foot.reset_index()

        # Round coordinates to avoid floating point issues
        xres, yres = footprint.xres, footprint.yres
        xdigits = StiltJacobianBuilder._calc_digits(xres)
        ydigits = StiltJacobianBuilder._calc_digits(yres)

        foot[x_dim] = foot[x_dim].round(xdigits)
        foot[y_dim] = foot[y_dim].round(ydigits)

        # Reorder dimensions to x, y, time
        foot = foot.set_index([x_dim, y_dim, 'time'])  # still a df

        # Build index value for the observation
        obs_index = StiltJacobianBuilder._build_obs_index(sim=sim)

        # Build Jacobian row for each set of coordinates
        rows = {}
        for key, coord_list in coords.items():
            # Round input coordinates to match footprint rounding
            coord_index = pd.MultiIndex.from_tuples(coord_list)
            coord_index = coord_index.round([xdigits, ydigits]).set_names([x_dim, y_dim])

            # Filter to xy points defined by grid
            filtered_foot = foot.reset_index(level='time').loc[coord_index].set_index('time', append=True)

            if filtered_foot.size == 0:
                continue

            # Integrate each simulation over flux_time bins
            integrated_foot = integrate_over_time_bins(data=filtered_foot,
                                                       time_bins=flux_times)

            # Transpose and set index as (obs_location, obs_time) multiindex
            transposed_foot = integrated_foot.T
            transposed_foot.index = obs_index

            rows[key] = transposed_foot
        return rows

    @staticmethod
    def _get_sim(simulation: stilt.Simulation | Path,
                 t_start: dt.datetime, t_stop: dt.datetime,
                 subset_hours: int | list[int] | None = None
                 ) -> stilt.Simulation | str | None:
        sim = StiltJacobianBuilder._load_simulation(simulation)

        if not sim.status == 'SUCCESS':
            return sim.id

        if not StiltJacobianBuilder._sim_in_time_range(sim=sim,
                                                       t_start=t_start, t_stop=t_stop,
                                                       subset_hours=subset_hours):
            return None
        return sim

    @staticmethod
    def _load_simulation(simulation: stilt.Simulation | Path) -> stilt.Simulation:
        if isinstance(simulation, Path):
            sim = stilt.Simulation.from_path(simulation)
        elif isinstance(simulation, stilt.Simulation):
            sim = simulation
        else:
            raise ValueError('simulation must be a Path or a stilt.Simulation object')
        return sim

    @staticmethod
    def _sim_in_time_range(sim: stilt.Simulation, t_start: dt.datetime, t_stop: dt.datetime,
                           subset_hours: int | list[int] | None = None) -> bool:
        n_hours = sim.config.n_hours
        if n_hours >= 0:
            raise ValueError('STILT must be run backwards in time (n_hours < 0)')
        n_hours = pd.to_timedelta(n_hours, unit='h')

        # Skip simulations that do not overlap with inversion time range
        sim_start = sim.receptor.time + n_hours
        sim_end = sim.receptor.time - pd.Timedelta(hours=1)
        if sim_end < t_start or sim_start >= t_stop:
            return False

        if subset_hours:
            # Subset simulations to specific hours
            if isinstance(subset_hours, int):
                subset_hours = [subset_hours]

            if not sim.receptor.time.hour in subset_hours:
                return False

        return True

    @staticmethod
    def _get_footprint(sim: stilt.Simulation,
                       t_start: dt.datetime, t_stop: dt.datetime,
                       resolution: str | None = None) -> stilt.Footprint | None:

        # Load footprint object
        footprint = StiltJacobianBuilder._load_footprint(sim=sim, resolution=resolution)

        if not footprint:
            return None

        # Check if footprint time range overlaps with inversion time range
        if not StiltJacobianBuilder._footprint_in_time_range(footprint=footprint,
                                                             t_start=t_start, t_stop=t_stop):
            # A footprint within a simulation might not overlap with the inversion time range
            # even if the simulation does
            return None

        return footprint

    @staticmethod
    def _load_footprint(sim: stilt.Simulation,
                        resolution: str | None = None) -> stilt.Footprint | None:
        if resolution:
            # Load footprint at specified resolution
            footprint = sim.footprints[resolution]
        else:
            # Get the default (highest) resolution footprint
            footprint = sim.footprint
        return footprint

    @staticmethod
    def _footprint_in_time_range(footprint: stilt.Footprint,
                               t_start: dt.datetime, t_stop: dt.datetime) -> bool:
        # Check if footprint time range overlaps with inversion time range
        return footprint.time_range[0] < t_stop and footprint.time_range[1] > t_start

    @staticmethod
    def _build_obs_index(sim: stilt.Simulation) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays([[sim.receptor.location.id], [sim.receptor.time]],
                                         names=['obs_location', 'obs_time'])

    @staticmethod
    def _calc_digits(res: float) -> int:
        if res <= 0:
            raise ValueError('Resolution must be positive')
        if res < 1:  # fractional resolution
            digits = int(np.ceil(np.abs(np.log10(res)))) + 1
        else:
            digits = int(-np.log10(res))
        return digits


class FluxInversion(InverseProblem):
    """
    FluxInversion: Atmospheric Flux Inversion Problem.

    Subclass of InverseProblem for estimating spatial and temporal surface fluxes
    (e.g., greenhouse gas emissions or uptake) from observed atmospheric concentrations.
    Combines observations, prior flux estimates, and a forward model (Jacobian)
    within a statistical estimation framework.

    Attributes
    ----------
    concentrations : pd.Series
        Observed concentrations used in the inversion.
    inventory : pd.Series
        Prior flux inventory.
    jacobian : pd.DataFrame
        Forward operator mapping fluxes to concentrations.
    prior_error : CovarianceMatrix
        Covariance matrix representing uncertainty in the prior flux inventory.
    modeldata_mismatch : CovarianceMatrix
        Covariance matrix representing uncertainty in observed concentrations and model-data mismatch.
    background : pd.Series, float, or None
        Background concentration.
    estimator : type[Estimator] or str, optional
        Estimation method or class to use for the inversion (e.g., 'bayesian'). Default is 'bayesian'.
    posterior_fluxes : pd.Series
        Estimated fluxes after inversion (posterior).
    posterior_concentrations : pd.Series
        Modelled concentrations using posterior fluxes.
    prior_concentrations : pd.Series
        Modelled concentrations using prior fluxes.
    plot : _Plotter
        Diagnostic and plotting interface.
    """

    def __init__(self,
                 concentrations: pd.Series,
                 inventory: pd.Series,
                 jacobian: Jacobian | pd.DataFrame,
                 prior_error: SymmetricMatrix,
                 modeldata_mismatch: SymmetricMatrix,
                 background: pd.Series | float | None = None,
                 estimator: type[Estimator] | str = 'bayesian',
                 **kwargs,
                 ) -> None:
        """
        Initialize a flux inversion problem.

        Parameters
        ----------
        concentrations : pd.Series
            Observed concentrations with a multi-index of (obs_location, obs_time).
        inventory : pd.Series
            Prior flux inventory with a multi-index of (time, lat, lon).
        jacobian : Jacobian | pd.DataFrame
            Jacobian matrix mapping fluxes to concentrations.
        prior_error : CovarianceMatrix
            Prior error covariance matrix.
        modeldata_mismatch : CovarianceMatrix
            Model-data mismatch covariance matrix.
        background : pd.Series | float | None, optional
            Background concentration to add to modelled concentrations, by default None.
        estimator : type[Estimator] | str, optional
            Estimator class or name to use for the inversion, by default 'bayesian'.
        kwargs : dict, optional
            Additional keyword arguments to pass to the InverseProblem constructor.
        """

        super().__init__(
            estimator=estimator,
            obs=concentrations,
            prior=inventory,
            forward_operator=jacobian,
            prior_error=prior_error,
            modeldata_mismatch=modeldata_mismatch,
            constant=background,
            **kwargs,
        )

        # Build plotting interface
        self.plot = _Plotter(self)

    @property
    def concentrations(self) -> pd.Series:
        return self.obs

    @property
    def inventory(self) -> pd.Series:
        return self.prior

    @property
    def jacobian(self) -> pd.DataFrame:
        return self.forward_operator

    @property
    def background(self) -> pd.Series | float | None:
        return self.constant

    @property
    def posterior_fluxes(self) -> pd.Series:
        return self.posterior

    @property
    def posterior_concentrations(self) -> pd.Series:
        return self.posterior_obs

    @property
    def prior_concentrations(self) -> pd.Series:
        return self.prior_obs


class _Plotter:
    """ Plotting interface for FluxInversion results."""

    def __init__(self, inversion: 'FluxInversion'):
        self.inversion = inversion

    def fluxes(self, time='mean', truth=None, **kwargs):
        """
        Plot prior & Posterior fluxes.

        Parameters
        ----------
        time : 'mean' | 'std' | int | pd.Timestamp, optional
            Time to plot. Can be 'mean' or 'std' to plot the mean or standard deviation
            over time, an integer to plot a specific time index, or a pd.Timestamp to plot a specific time.
            By default 'mean'.
        tiler : cartopy.io.img_tiles.GoogleTiles | None, optional
            Tiler to use for background map, by default None.
            If provided, the tiler will be used to add a background map to the plots.
        truth : pd.Series | None, optional
            Truth fluxes to plot for comparison, by default None.
            Residual will be calculated as posterior - truth if provided,
            otherwise as posterior - prior.
        **kwargs : dict
            Additional keyword arguments to pass to xarray plotting functions.

        Returns
        -------
        fig, axes : matplotlib.figure.Figure, np.ndarray
            Figure and axes objects.
        """
        # Get xarray representations of fluxes
        prior = self.inversion.xr.prior
        posterior = self.inversion.xr.posterior_fluxes

        # Filter/aggregate by time
        if time == 'mean':
            prior = prior.mean(dim='time')
            posterior = posterior.mean(dim='time')
        elif time == 'std':
            prior = prior.std(dim='time')
            posterior = posterior.std(dim='time')
        elif isinstance(time, int):
            prior = prior.isel(time=time)
            posterior = posterior.isel(time=time)
        else:
            prior = prior.sel(time=time)
            posterior = posterior.sel(time=time)

        # Get tiler and projection from kwargs
        tiler = kwargs.pop('tiler', None)
        subplot_kw = kwargs.pop('subplot_kw', {})
        if tiler is not None:
            subplot_kw['projection'] = tiler.crs

        ncols = 3
        if time == 'std':
            ncols -= 1
        if truth is not None:
            ncols += 1
            if isinstance(truth, pd.Series):
                truth = truth.to_xarray()
            if time == 'mean':
                truth = truth.mean(dim='time')
            elif time == 'std':
                truth = truth.std(dim='time')
            elif isinstance(time, int):
                truth = truth.isel(time=time)
            else:
                truth = truth.sel(time=time)

        # Create figure and axes
        fig, axes = plt.subplots(ncols=ncols, sharey=True,
                                 subplot_kw=subplot_kw)

        if truth is None:
            ax_prior = axes[0]
            ax_post = axes[1]
        else:
            ax_truth = axes[0]
            ax_prior = axes[1]
            ax_post = axes[2]
        if time != 'std':
            ax_res = axes[-1]

        # Add background tiles
        if tiler is not None:
            tiler_zoom = kwargs.pop('tiler_zoom', 10)
            extent = [posterior.lon.min(), posterior.lon.max(),
                    posterior.lat.min(), posterior.lat.max()]
            for ax in axes:
                ax.set_extent(extent, crs=PC)
                ax.add_image(tiler, tiler_zoom)
            if 'lat' in posterior.dims:
                kwargs['transform'] = PC
            else:
                # TODO handle projected data (i could use my crs class)
                raise ValueError('Cannot determine coordinate reference system for plotting.')

        # Colorbar and plot options
        vmin = min(prior.min(), posterior.min())
        vmax = max(prior.max(), posterior.max())
        if truth is not None:
            vmin = min(vmin, truth.min())
            vmax = max(vmax, truth.max())
        alpha = kwargs.pop('alpha', 0.55)
        cmap = kwargs.pop('cmap', 'RdBu_r' if vmin < 0 else 'Reds')
        if vmin < 0:
            center = 0
            vmin = None  # cant set both vmin/vmax and center
        else:
            center = None

        # Set colorbar axis below both plots
        fig.subplots_adjust(bottom=0.15)
        ax1 = axes[0]
        cbar_ax1_width = ax_post.get_position().x1 - ax1.get_position().x0
        cbar_ax1 = fig.add_axes([ax1.get_position().x0, ax1.get_position().y0 - 0.1, cbar_ax1_width, 0.05])
        
        if truth is not None:
            truth.plot(ax=ax_truth, x='lon', y='lat', vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha,
                       add_colorbar=False, center=center, **kwargs)
            ax_truth.set(title='Truth',
                         xlabel=None,
                         ylabel=None)

        prior.plot(ax=ax_prior, x='lon', y='lat', vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha,
                       cbar_ax=cbar_ax1, cbar_kwargs={'orientation': 'horizontal', 'label': 'Flux'},
                       center=center, **kwargs)
        posterior.plot(ax=ax_post, x='lon', y='lat', vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha,
                       add_colorbar=False, center=center, **kwargs)

        # Add title and time text
        fig.suptitle("Flux Maps", fontsize=16, y=ax_prior.get_position().y1 + 0.13)
        fig.text(0.5, ax_prior.get_position().y1 + 0.05, f"time = {time}", ha='center', va='bottom', fontsize=10)

        # Set labels for each subplot
        ax_prior.set(title='Prior',
                     xlabel=None,
                     ylabel=None)
        ax_post.set(title='Posterior',
                    xlabel=None,
                    ylabel=None)

        # Plot residual
        if time != 'std':
            if truth is not None:
                base = truth
                label = 'Posterior - Truth'
            else:
                base = prior
                label = 'Posterior - Prior'
            residual_cmap = kwargs.pop('residual_cmap', 'PiYG')
            cbar_ax2 = fig.add_axes([ax_res.get_position().x0, ax_res.get_position().y0 - 0.1, ax_res.get_position().width, 0.05])
            (posterior - base).plot(ax=ax_res, x='lon', y='lat', cmap=residual_cmap, alpha=alpha, center=0,
                                         cbar_ax=cbar_ax2, cbar_kwargs={'orientation': 'horizontal',
                                                                        'label': label},
                                         **kwargs)

            ax_res.set(title='Residual',
                       xlabel=None,
                       ylabel=None)

        return fig, axes

    def concentrations(self, location=None, **kwargs):
        """
        Plot observed, prior, & posterior concentrations.
        
        Parameters
        ----------
        location : str | list[str] | None, optional
            Observation location(s) to plot. If None, plots all locations.
            By default None.
        **kwargs : dict
            Additional keyword arguments to pass to pandas plotting functions.

        Returns
        -------
        axes : list[matplotlib.axes.Axes]
            List of axes objects.
        """
        obs = self.inversion.concentrations
        posterior = self.inversion.posterior_concentrations
        prior = self.inversion.prior_concentrations

        data = pd.concat([obs, posterior, prior], axis=1)

        if location is None:
            locations = data.index.get_level_values('obs_location').unique()
        elif isinstance(location, str):
            locations = [location]
        elif isinstance(location, list):
            locations = location
        else:
            raise ValueError('location must be None, a string, or a list of strings')

        axes = []
        for location in locations:
            df = data.loc[location]
            df.columns.name = None

            fig, ax = plt.subplots()

            df.plot(ax=ax, style='.', alpha=0.6, color=['black', 'red', 'blue'], markeredgecolor='None', legend=False)
            df.rolling(window=max(1, int(len(df)/10)), center=True).mean().plot(ax=ax, linewidth=2,
                                                                                  color=['black', 'red', 'blue'],
                                                                                  label=['Observed', 'Posterior', 'Prior'],)
            ax.set(title=f'Concentrations at {location}', ylabel='Concentration', xlabel='Time')
            fig.autofmt_xdate()
            axes.append(ax)
            plt.show()
        return axes