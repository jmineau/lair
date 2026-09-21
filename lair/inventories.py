"""
Emissions Inventories

An emission(s) inventory is an accounting of the amount of pollutants
discharged into the atmosphere. An emission inventory usually contains
the total emissions for one or more specific greenhouse gases or air
pollutants, originating from all source categories in a certain
geographical area and within a specified time span, usually a specific year.
"""

from __future__ import annotations  # keep optional-dep annotations (e.g. shapely Polygon) lazy

import datetime as dt
import os
import re
from abc import ABCMeta
from pathlib import Path
from typing import Any, Literal, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import xarray as xr
from typing_extensions import \
    Self  # requires python 3.11 to import from typing
from xarray import DataArray, Dataset

from lair import units
from lair.config import get_data_dir
from lair.geo import (CRS, PC, BaseGrid, round_latlon, wrap_lons,
                            write_rio_crs)
from lair._optional import import_optional_dependency

# Optional dependency for chemistry calculations
molmass = import_optional_dependency("molmass")
shapely = import_optional_dependency("shapely")
from molmass import Formula  # noqa: E402
from shapely import Polygon  # noqa: E402

xr.set_options(keep_attrs=True)


#: Environment variable holding the inventory archive root
#: (with EDGAR/, EPA/, GFEI/, vulcan/ and WetCHARTs/ subdirectories)
INVENTORY_DIR_ENV = 'LAIR_INVENTORY_DIR'

#: Default destination units
DST_UNITS: str = 'kg km-2 d-1'

DEFAULT_PINT_FMT = '~C'

Regrid_Methods = Literal['conservative', 'conservative_normed']

_XarrayT = TypeVar('_XarrayT', bound=DataArray | Dataset)


#: Pollutants whose emissions are reported as the mass of another species
_MASS_BASIS = {'NOX': 'NO2'}  # NOx is conventionally reported as NO2 mass


def molecular_weight(pollutant: str) -> pint.Quantity:
    """
    Calculate the molecular weight of a pollutant.

    NOx is taken as NO2, the usual reporting basis for NOx emissions.

    Parameters
    ----------
    pollutant : str
        The pollutant, as a chemical formula (case-sensitive, e.g. ``'CH4'``).

    Returns
    -------
    pint.Quantity
        The molecular weight.
    """
    formula = _MASS_BASIS.get(pollutant.upper(), pollutant)
    return Formula(formula).mass * units('g/mol')


def convert_units(data: _XarrayT, pollutant: str, dst_units: Any) -> _XarrayT:
    """
    Convert the units of a pint quantified data array or dataset.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        The data to convert. Must have pint units.
    pollutant : str
        The pollutant.
    dst_units : Any
        The destination units.

    Returns
    -------
    xr.DataArray | xr.Dataset
        The data with converted units.
    """
    data = data.copy()

    # Calculate molecular weight of pollutant
    mw = molecular_weight(pollutant)

    # Use custom pint context to convert mass <--> substance
    with units.context('mass_flux', mw=mw):
        if isinstance(data, xr.Dataset):
            data = data.pint.to(dst_units)
            for var in data.data_vars:
                data[var].attrs['units'] = dst_units
        elif isinstance(data, xr.DataArray):
            data = data.pint.to(dst_units)
            data.attrs['units'] = dst_units
        else:
            raise TypeError("data must be an xarray Dataset or DataArray")
    return data


def sum_sectors(data: Dataset) -> DataArray:
    """
    Sum emissions from all sectors in the dataset.

    Parameters
    ----------
    data : xr.Dataset
        The inventory data with emissions from multiple sectors as different variables.

    Returns
    -------
    xr.DataArray
        The sum of emissions from all sectors.
    """
    total = data.to_array(dim='sector', name='emissions').sum('sector')
    total.attrs['long_name'] = 'Total Emissions'
    total.attrs['units'] = data[list(data.data_vars)[0]].attrs['units']

    return total


class Inventory(BaseGrid):
    """
    Base class for inventories.
    """

    def __init__(self,
                 data: str | Path | Dataset,
                 pollutant: str,
                 src_units: str | None = None,
                 time_step: str = 'annual',
                 crs: str = 'EPSG:4326',
                 version: str | None = None,
                 standardize_units: bool = False,
                 ) -> None:
        """
        Initialize the inventory.

        Parameters
        ----------
        data : str | Path | xr.Dataset
            The inventory data. If a string or Path, the path to the data. If an xr.Dataset, the data itself.
            Data should have dimensions of time, lat, lon and each variable should be a different emissions sector.
            If the source data is not in the correct format, the `_process` method should be overridden.
        pollutant : str
            The pollutant.
        src_units : str, optional
            The source units of the data, by default None. If None, the units are extracted from the data attributes.
        time_step : str, optional
            The time step of the data, by default 'annual'.
        crs : str, optional
            The CRS of the data, by default 'EPSG:4326'.
        version : str, optional
            The version of the inventory, by default None.
        standardize_units : bool, optional
            Whether to standardize the units of all variables to the same units, by default False.
            If True, all variables are assumed to be emissions and in the same units.
            If False, the units of each variable are retained as-is.
        """

        # Upper-case all-lowercase names ('ch4' -> 'CH4') but keep mixed-case
        # formulas as given: 'NOx'.upper() would no longer be a formula
        self.pollutant: str = (pollutant if any(c.isupper() for c in pollutant)
                               else pollutant.upper())
        self.time_step: str = time_step
        self.crs = CRS(crs)
        self.version: str | None = version

        self.path: str | None
        if isinstance(data, str | Path):
            self.path = str(data)

            # Open dataset
            files = self.get_files()
            if not files:
                raise FileNotFoundError(f'No {type(self).__name__} files found under {self.path}')
            ds = self._open(files)

            # Apply inventory-specific processing
            ds = self._process(ds)
        elif isinstance(data, Dataset):
            self.path = None
            ds = data
            if src_units is None:
                var = list(ds.data_vars)[0]
                src_units = ds[var].attrs.get('units')
        else:
            raise ValueError('Data must be a path to a file or an xarray Dataset')
        if src_units is None:
            raise ValueError('Units must be provided in the data attributes or as an argument')

        # Standardize units
        # - Requirements:
        #   - all variables are emissions
        #   - all in the same units
        self.src_units: str | pint.registry.Unit = src_units
        ds = self._quantify(ds)  # quantify using provided src_units
        if standardize_units:
            ds = convert_units(ds, pollutant=self.pollutant, dst_units=DST_UNITS)

        # Set the rioxarray CRS. Projected inventories (e.g. Vulcan) are on x/y
        # with 2D lat/lon coords; the rest are on 1D lat/lon.
        x_dim, y_dim = ('lon', 'lat') if 'lon' in ds.dims else ('x', 'y')
        ds = ds.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
        ds = write_rio_crs(ds, self.crs)

        # Store the data
        self._data: Dataset = ds

    def get_standard_name(self) -> str:
        """
        Get the standard name of the inventory.

        Returns
        -------
        str
            The standard name.
        """
        return f'{self.time_step.lower()}_{self.pollutant}_emissions'

    def get_files(self) -> None | list[Path]:
        """
        Get the inventory files.

        Returns
        -------
        None | list[Path]
            The inventory files.
        """
        if self.path is None:
            return None
        p = Path(self.path)
        if p.is_file():
            return [p]
        else :
            return list(p.glob('*.nc'))

    def _file_root(self) -> Path:
        """The inventory path, for subclasses that find their own files."""
        if self.path is None:
            raise ValueError(f'This {type(self).__name__} was built from a Dataset and has no files.')
        return Path(self.path)

    def get_units(self, pint: bool = False) -> tuple[Any, Any, Any]:
        """
        Get the quantity, area, and time units of the inventory data.

        Parameters
        ----------
        pint : bool, optional
            Whether to return `pint` units, by default False.

        Returns
        -------
        tuple[Any, Any, Any]
            The quantity, area, and time units.
        """
        # All variables should be in the same units
        var = list(self._data.data_vars)[0]
        data_units = f'{self._data[var].pint.units: ~C}'  # compact symbols like "kg/km**2/day"
        quantity_unit, area_unit, time_unit = data_units.split('/')

        if pint:
            return units(quantity_unit), units(area_unit), units(time_unit)
        else:
            return quantity_unit, area_unit, time_unit

    @property
    def absolute_emissions(self) -> Dataset:
        """
        Calculate the absolute emissions (total per gridcell for time step by variable).

        Returns
        -------
        xr.DataArray
            The absolute emissions.
        """
        # Extract current unit information
        _, area_unit, time_unit = self.get_units(pint=False)

        # Multiply by the gridcell area to get mass|substance per time per time step
        absolute = self._data * (self.gridcell_area * units('km**2')).pint.to(area_unit)

        # Get the number of seconds in each time step
        # - I am calculating the exact number of seconds in each time step.
        #   Inventory providers may have used a simpler method of avg secs per time step.
        #   However, its probably close enough to not matter  TODO check this
        time = self._data.time
        if self.time_step == 'annual':
            seconds_per_step = (time.dt.is_leap_year * 366 + (~time.dt.is_leap_year) * 365) * 24 * 3600
        elif self.time_step == 'monthly':
            seconds_per_step = time.dt.days_in_month * 24 * 3600
        elif self.time_step == 'quarterly':
            t = time.to_index()  # days in each 3-month quarter starting at `time`
            seconds_per_step = ((t + pd.DateOffset(months=3)) - t).days.to_numpy() * 24 * 3600
        elif self.time_step == 'biweekly':
            seconds_per_step = np.full(time.size, 14 * 24 * 3600)
        elif self.time_step == 'daily':
            seconds_per_step = np.full(time.size, 24 * 3600)
        elif self.time_step == 'hourly':
            seconds_per_step = np.full(time.size, 3600)
        else:
            raise ValueError(f'Time step {self.time_step} not supported')
        if isinstance(seconds_per_step, xr.DataArray):
            seconds_per_step = seconds_per_step.data
        seconds_per_step = self._data.assign(sec_per_step=('time', seconds_per_step)).sec_per_step

        # Then multiply by the time in the time step to get mass|substance per gridcell
        absolute = absolute * (seconds_per_step * units('s')).pint.to(time_unit)

        absolute.attrs = {'long_name': 'Absolute Emissions',
                          'standard_name': f'{self.time_step.lower()}_emissions_per_gridcell'}
        return absolute.pint.dequantify(DEFAULT_PINT_FMT)

    @property
    def total_emissions(self) -> DataArray:
        """
        Calculate the total emissions by summation over all variables.

        Returns
        -------
        xr.DataArray
            The total emissions.
        """
        return sum_sectors(self.data)

    @property
    def collapsed(self) -> DataArray:
        """
        Collapse the inventory data to a single variable.

        Returns
        -------
        xr.DataArray
            The collapsed data.
        """
        return self._data.to_dataarray(dim='sector', name='emissions').pint.dequantify(DEFAULT_PINT_FMT)

    # A property here, a plain attribute on BaseGrid
    @property
    def data(self) -> Dataset:  # pyrefly: ignore[bad-override]
        """
        The inventory data.

        .. note::
            `pint` units have been dequantified and stored in variable attributes.

        Returns
        -------
        xr.DataArray | xr.Dataset
            The inventory data.
        """
        return self._data.pint.dequantify(DEFAULT_PINT_FMT)

    @data.setter
    def data(self, data: Dataset) -> None:
        # Reattach pint units to the data from attrs
        self._data = data.pint.quantify()

    def quantify(self) -> Dataset:
        """
        Quantify the data using `pint` units for each variable.

        Returns
        -------
        xr.Dataset
            The quantified data.
        """
        return self._data

    def convert_units(self, dst_units: str, inplace: bool = False) -> Self:
        """
        Convert the units of the data to the desired output units.

        Parameters
        ----------
        dst_units : Any
            The destination units.
        inplace : bool, optional
            Whether to modify the data in place, by default False.

        Returns
        -------
        Inventory
            The inventory with converted units. If `inplace=True` returns `self`,
            otherwise returns a new Inventory copy.
        """
        data = convert_units(data=self._data, pollutant=self.pollutant, dst_units=dst_units)
        if inplace:
            self._data = data
            self.src_units = dst_units
            return self
        else:
            new = self.copy()
            new._data = data
            new.src_units = dst_units
            return new

    def integrate(self) -> DataArray:
        """
        Integrate the data over the spatial dimensions
        to get the total emissions per time step.

        Returns
        -------
        xr.DataArray
            The integrated data.
        """
        x_dim, y_dim = self.data.rio.x_dim, self.data.rio.y_dim
        return sum_sectors(self.absolute_emissions.sum([x_dim, y_dim]))

    # Emission inventories only allow mass-conserving (conservative) regridding
    # pyrefly: ignore[bad-override]
    def regrid(self, out_grid: Dataset,
               method: Regrid_Methods = 'conservative', inplace: bool = False) -> Self:
        """
        Regrid the data to a new grid. Uses `xesmf` for regridding.

        .. note::
            At present, `xesmf` only supports regridding lat-lon grids. self.data must be on a lat-lon grid.
            Possibly could use `xesmf.frontend.BaseRegridder` to regrid to a generic grid.

        .. warning::
            `xarray.Dataset.cf.add_bounds` is known to have issues, including near the 180th meridian.
            Care should be taken when using this method, especially with global datasets.

        Parameters
        ----------
        out_grid : xr.DataArray
            The new grid to resample to. Must be a lat-lon grid.
        method : str, optional
            The regridding method, by default 'conservative'.

            .. note::
                Other `xesmf` regrid methods can be passed,
                but it is **highly** encouraged to use a conservative method for fluxes.
        inplace : bool, optional
            Whether to modify the data in place, by default False.

        Returns
        -------
        Inventory
            The regridded inventory. If `inplace=True` returns `self`,
            otherwise returns a new Inventory copy.
        """
        return super().regrid(out_grid, method=method, inplace=inplace)

    # Emission inventories only allow mass-conserving (conservative) regridding
    # pyrefly: ignore[bad-override]
    def resample(self, resolution: float | tuple[float, float],
                 regrid_method: Regrid_Methods = 'conservative', inplace: bool = False) -> Self:
        """
        Resample the data to a new resolution.

        Parameters
        ----------
        resolution : float | tuple[x_res, y_res]
            The new resolution in degrees. If a single value is provided, the resolution
            is assumed to be the same in both dimensions.
        regrid_method : str, optional
            The regridding method, by default 'conservative'.

        Returns
        -------
        BaseGrid
            The resampled grid
        """
        return super().resample(resolution, regrid_method, inplace=inplace)

    # Emission inventories only allow mass-conserving (conservative) regridding
    # pyrefly: ignore[bad-override]
    def reproject(self, resolution: float | tuple[float, float],
                  regrid_method: Regrid_Methods = 'conservative', inplace: bool = False) -> Self:
        """
        Reproject the data to a lat lon rectilinear grid.

        Parameters
        ----------
        resolution : float | tuple[x_res, y_res]
            The new resolution in degrees. If a single value is provided, the resolution
            is assumed to be the same in both dimensions.
        regrid_method : str, optional
            The regridding method, by default 'conservative'.

        Returns
        -------
        BaseGrid
            The reprojected grid
        """
        return super().reproject(resolution, regrid_method, inplace=inplace)

    def plot(self, ax: plt.Axes | None = None,
             time: str | int = 'mean',
             sector: str | None = None, **kwargs) -> plt.Axes:
        """
        Plot the inventory data.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on.
        time : str | int, optional
            The time step to plot, by default 'mean'. If 'mean', the mean emissions are plotted.
            Otherwise pass a string selector or integer index.
        sector : str, optional
            The sector to plot, by default None. If None, the total emissions are plotted.
        kwargs : dict
            Additional keyword arguments to pass to `xarray`'s plot method.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': PC})

        if sector is not None:
            data = self.data[sector]
        else:
            data = self.total_emissions

        if time == 'mean':
            data = data.mean('time')
        elif isinstance(time, int):
            data = data.isel(time=time)
        else:
            data = data.sel(time=time)

        data.plot(ax=ax, x='lon', y='lat', transform=PC, **kwargs)

        return ax

    def _open(self, files: list[Path]) -> Dataset:
        # Open the dataset, preprocessing if necessary
        return xr.open_mfdataset(files, preprocess=getattr(self, '_preprocess', None))

    def _process(self, data) -> Dataset:
        # In the base case, we just return the data
        # This method can be overridden in the subclasses to process the data
        # Resulting ds hould have coords of time, lat, lon and all variables should be emissions
        return data

    def _quantify(self, data: Dataset) -> Dataset:
        # Quantify the entire dataset at once to keep coord indexes consistent.
        units_map = {var: self.src_units for var in data.data_vars}
        return data.pint.quantify(units_map)


class MultiModelInventory(Inventory):
    """
    Base class for inventories that are multi-model.
    """
    multimodel_data: Dataset

    def __init__(self,
                 data: str | Path | Dataset,
                 pollutant: str,
                 src_units: str | None = None,
                 time_step: str = 'annual',
                 crs: str = 'EPSG:4326',
                 version: str | None = None,
                 model: str|None = None) -> None:
        """
        Initialize the multi-model inventory.

        Parameters
        ----------
        data : str | Path | xr.Dataset
            The inventory data. If a string or Path, the path to the data. If an xr.Dataset, the data itself.
            Data should have dimensions of time, lat, lon and each variable should be a different emissions sector.
            If the source data is not in the correct format, the `_process` method should be overridden.
        pollutant : str
            The pollutant.
        src_units : str, optional
            The source units of the data, by default None. If None, the units are extracted from the data attributes.
        time_step : str, optional
            The time step of the data, by default 'annual'.
        crs : str, optional
            The CRS of the data, by default 'EPSG:4326'.
        version : str, optional
            The version of the inventory, by default None.
        model : str, optional
            The model to select from the multimodel data, by default None.
            If None, the mean of all models is used.
        """
        self.model = model or 'mean'
        super().__init__(data, pollutant, src_units=src_units,
                         time_step=time_step, crs=crs, version=version)

    def _process(self, data: Dataset) -> Dataset:
        self.multimodel_data = data

        if self.model == 'mean':
            return data.mean(dim='model')
        elif self.model == 'median':
            return data.median(dim='model')
        else:
            return data.sel(model=self.model)


class EDGAR(Inventory, metaclass=ABCMeta):
    """
    EDGAR - Emissions Database for Global Atmospheric Research

    https://edgar.jrc.ec.europa.eu/

    EDGAR is a multipurpose, independent, global database of anthropogenic
    emissions of greenhouse gases and air pollution on Earth. EDGAR provides
    independent emission estimates compared to what reported by European Member
    States or by Parties under the United Nations Framework Convention on Climate
    Change (UNFCCC), using international statistics and a consistent IPCC methodology.

    EDGAR provides both emissions as national totals and gridmaps at 0.1 x 0.1 degree
    resolution at global level, with yearly, monthly and up to hourly data. 
    """
    src_units: str = 'kg m-2 s-1'

    sectors = {
        "AGS": {
            "description": "Agricultural soils",
            "IPCC_1996_code": "4C+4D1+4D2+4D4",
            "IPCC_2006_code": "3C2+3C3+3C4+3C7"
        },
        "AWB": {
            "description": "Agricultural waste burning",
            "IPCC_1996_code": "4F",
            "IPCC_2006_code": "3C1b"
        },
        "CHE": {
            "description": "Chemical processes",
            "IPCC_1996_code": "2B",
            "IPCC_2006_code": "2B"
        },
        "ENE": {
            "description": "Power industry",
            "IPCC_1996_code": "1A1a",
            "IPCC_2006_code": "1A1a"
        },
        "ENF": {
            "description": "Enteric fermentation",
            "IPCC_1996_code": "4A",
            "IPCC_2006_code": "3A1"
        },
        "FFF": {
            "description": "Fossil Fuel Fires",
            "IPCC_1996_code": "7A",
            "IPCC_2006_code": "5B"
        },
        "IDE": {
            "description": "Indirect emissions from NOx and NH3",
            "IPCC_1996_code": "7B+7C",
            "IPCC_2006_code": "5A"
        },
        "IND": {
            "description": "Combustion for manufacturing",
            "IPCC_1996_code": "1A2",
            "IPCC_2006_code": "1A2"
        },
        "IRO": {
            "description": "Iron and steel production",
            "IPCC_1996_code": "2C1a+2C1c+2C1d+2C1e+2C1f+2C2",
            "IPCC_2006_code": "2C1+2C2"
        },
        "MNM": {
            "description": "Manure management",
            "IPCC_1996_code": "4B",
            "IPCC_2006_code": "3A2"
        },
        "N2O": {
            "description": "Indirect N2O emissions from agriculture",
            "IPCC_1996_code": "4D3",
            "IPCC_2006_code": "3C5+3C6"
        },
        "PRO": {
            "description": "Fuel exploitation",
            "IPCC_1996_code": "1B1a+1B2a1+1B2a2+1B2a3+1B2a4+1B2c",
            "IPCC_2006_code": "1B1a+1B2aiii2+1B2aiii3+1B2bi+1B2bii"
        },
        "PRO_COAL": {
            "description": "Fuel exploitation COAL",
            "IPCC_1996_code": "1B1a",
            "IPCC_2006_code": "1B1a"
        },
        "PRO_FFF": {
            "description": "Fuel exploitation (including fossil fuel fires)",
            "IPCC_1996_code": "1B1a+1B2a1+1B2a2+1B2a3+1B2a4+1B2c+7A",
            "IPCC_2006_code": "1B1a+1B2aiii2+1B2aiii3+1B2bi+1B2bii+5B"
        },
        "PRO_GAS": {
            "description": "Fuel exploitation GAS",
            "IPCC_1996_code": "1B2c",
            "IPCC_2006_code": "1B2bi+1B2bii"
        },
        "PRO_OIL": {
            "description": "Fuel exploitation OIL",
            "IPCC_1996_code": "1B2a1+1B2a2+1B2a3+1B2a4",
            "IPCC_2006_code": "1B2aiii2+1B2aiii3"
        },
        "PRU_SOL": {
            "description": "Solvents and products use",
            "IPCC_1996_code": "3",
            "IPCC_2006_code": "2D3+2E+2F+2G"
        },
        "RCO": {
            "description": "Energy for buildings",
            "IPCC_1996_code": "1A4",
            "IPCC_2006_code": "1A4+1A5"
        },
        "REF_TRF": {
            "description": "Oil refineries and Transformation industry",
            "IPCC_1996_code": "1A1b+1A1c+1A5b1+1B1b+1B2a5+1B2a6+1B2b5+2C1b",
            "IPCC_2006_code": "1A1b+1A1ci+1A1cii+1A5biii+1B1b+1B2aiii6+1B2biii3+1B1c"
        },
        "SWD_INC": {
            "description": "Solid waste incineration",
            "IPCC_1996_code": "6C+6Dhaz",
            "IPCC_2006_code": "4C"
        },
        "SWD_LDF": {
            "description": "Solid waste landfills",
            "IPCC_1996_code": "6A+6Dcom",
            "IPCC_2006_code": "4A+4B"
        },
        "TNR_Aviation_CDS": {
            "description": "Aviation climbing&descent",
            "IPCC_1996_code": "1A3a_CDS",
            "IPCC_2006_code": "1A3a_CDS"
        },
        "TNR_Aviation_CRS": {
            "description": "Aviation cruise",
            "IPCC_1996_code": "1A3a_CRS",
            "IPCC_2006_code": "1A3a_CRS"
        },
        "TNR_Aviation_LTO": {
            "description": "Aviation landing&takeoff",
            "IPCC_1996_code": "1A3a_LTO",
            "IPCC_2006_code": "1A3a_LTO"
        },
        "TNR_Other": {
            "description": "Railways, pipelines, off-road transport",
            "IPCC_1996_code": "1A3c+1A3e",
            "IPCC_2006_code": "1A3c+1A3e"
        },
        "TNR_Ship": {
            "description": "Shipping",
            "IPCC_1996_code": "1A3d+1C2",
            "IPCC_2006_code": "1A3d"
        },
        "TRO": {
            "description": "Road transportation",
            "IPCC_1996_code": "1A3b",
            "IPCC_2006_code": "1A3b"
        },
        "TRO_noRES": {
            "description": "Road transportation no resuspension",
            "IPCC_1996_code": "1A3b_noRES",
            "IPCC_2006_code": "1A3b_noRES"
        },
        "WWT": {
            "description": "Waste water handling",
            "IPCC_1996_code": "6B",
            "IPCC_2006_code": "4D"
        },
    }

    def get_files(self) -> list[Path]:
        """
        Recursively get the inventory files for each sector.

        Returns
        -------
        list[Path]
            The inventory files.
        """
        return [f for f in self._file_root().rglob('*.nc')
                if 'TOTALS' not in f.stem]

    def get_sector_name(self, sector: str) -> str:
        """
        Generate a formatted name from the sector description.

        Parameters
        ----------
        sector : str
            The key of the sector in the sectors dictionary.

        Returns
        -------
        str
            The formatted name.
        """
        # Retrieve the description from the sectors dictionary
        description = self.sectors[sector]['description']
        # Substitute unwanted characters in the 'description' key to create 'name'
        name = re.sub(r'[\s\(\&\-]', '_', description)\
            .replace(')', '').replace(',', '')
        return name


class EDGARv7(EDGAR):
    """
    EDGAR v7 - Global Greenhouse Gas Emissions

    https://edgar.jrc.ec.europa.eu/dataset_ghg70

    EDGAR (Emissions Database for Global Atmospheric Research) Community
    GHG Database (a collaboration between the European Commission, Joint
    Research Centre (JRC), the International Energy Agency (IEA), and
    comprising IEA-EDGAR CO2, EDGAR CH4, EDGAR N2O, EDGAR F-GASES version 7.0,
    (2022) European Commission, JRC (Datasets).
    """
    version: str = 'v7'

    def __init__(self, pollutant: str, inventory_dir: str | None = None) -> None:
        """
        Initialize the EDGAR inventory.

        Parameters
        ----------
        pollutant : str
            The pollutant.
        inventory_dir : str, optional
            Root of the inventory archive, by default ``$LAIR_INVENTORY_DIR``.
        """
        self.edgar_dir = os.path.join(get_data_dir(INVENTORY_DIR_ENV, inventory_dir), 'EDGAR')
        path = os.path.join(self.edgar_dir, self.version, pollutant)
        super().__init__(path, pollutant,
                         src_units=self.src_units, version=self.version)

    def _preprocess(self, ds: Dataset) -> Dataset | None:
        # Need to strip year and sector name from filename
        filename = ds.encoding['source']

        pattern = r'_(\d{4})_([\w_]+)\.0\.1x0\.1\.nc$'
        match = re.search(pattern, filename)

        if match:
            # Extract the year and the text between the year and flx.nc
            year = match.group(1)
            sector_code = match.group(2)
        else:
            return None

        var = self.get_sector_name(sector_code)

        ds = ds.rename({'emi_ch4': var})
        ds[var].attrs['long_name'] = f'{var}_Emissions'
        ds[var].attrs['standard_name'] = self.get_standard_name()

        # Add time coordinate
        ds = ds.expand_dims(time=[dt.datetime(int(year), 1, 1)])
        return ds

    def _process(self, data: Dataset) -> Dataset:
        data = data.assign_coords(lon=wrap_lons(data.lon))\
            .sortby('lon')

        # Round grid to nearest 0.1 degrees
        # - cell coordinates are center-of-cell, so we actually need to round to nearest 0.05
        data = round_latlon(data, 2, 2)
        return data


class EDGARv8(EDGAR):
    """
    EDGAR v8 - Global Greenhouse Gas Emissions

    https://edgar.jrc.ec.europa.eu/dataset_ghg80

    EDGAR (Emissions Database for Global Atmospheric Research) Community
    GHG Database, a collaboration between the European Commission, Joint
    Research Centre (JRC), the International Energy Agency (IEA), and
    comprising IEA-EDGAR CO2, EDGAR CH4, EDGAR N2O, EDGAR F-GASES
    version 8.0, (2023) European Commission, JRC (Datasets).
    """
    version: str = 'v8'

    def __init__(self, pollutant: str, time_step: Literal['annual', 'monthly']='annual',
                 inventory_dir: str | None = None):
        """
        Initialize the EDGAR inventory.

        Parameters
        ----------
        pollutant : str
            The pollutant.
        time_step : Literal['annual', 'monthly'], optional
            The time step of the data, by default 'annual'.
        inventory_dir : str, optional
            Root of the inventory archive, by default ``$LAIR_INVENTORY_DIR``.
        """
        self.edgar_dir = os.path.join(get_data_dir(INVENTORY_DIR_ENV, inventory_dir), 'EDGAR')
        path = os.path.join(self.edgar_dir, self.version,
                            '' if time_step == 'annual' else time_step, pollutant)
        super().__init__(path, pollutant,
                         src_units=self.src_units, time_step=time_step, version=self.version)

    def _preprocess(self, ds: Dataset) -> Dataset:
        # Rename var and add attributes
        old_var = 'fluxes'
        var = ds[old_var].attrs['long_name'].replace(' ', '_').replace(',', '').replace('-', '_')
        ds = ds.rename({old_var: var})
        attrs = ds[var].attrs
        attrs['long_name'] = f'{var}_Emissions'
        attrs['standard_name'] = self.get_standard_name()

        if self.time_step == 'annual':
            # Add time coordinate to annual data
            ds = ds.expand_dims(time=[dt.datetime(int(attrs['year']), 1, 1)])
        return ds

    def _process(self, data: Dataset) -> Dataset:
        # Fuel_exploitation is the sum of Fuel_exploitation_COAL, Fuel_exploitation_GAS, and Fuel_exploitation_OIL
        data = data.drop_vars(['Fuel_exploitation'], errors='ignore')
        return data


class EPA(Inventory, metaclass=ABCMeta):
    """
    United States Environmental Protection Agency (EPA) Gridded Methane Emissions

    https://www.epa.gov/ghgemissions/us-gridded-methane-emissions

    The gridded EPA U.S. methane greenhouse gas inventory (gridded methane GHGI)
    includes time series of annual methane emission maps with 0.1° x 0.1°
    (~ 10 x 10 km) spatial and monthly temporal resolution for the contiguous
    United State (CONUS). This gridded methane inventory is designed to be
    consistent with methane emissions from the U.S. EPA Inventory of U.S.
    Greenhouse Gas Emissions and Sinks (U.S. GHGI).
    """
    pollutant: str = 'CH4'
    src_units: str = 'molec cm-2 s-1'

    _emissions_prefix: str
    _variable_pattern: str = r'{}(?:_Supp)?_([1-9][A-Z]?[1-9]?[a-z]*)_([A-Za-z_]*)'
    _lat_deci = 2
    _lon_deci = 2

    def _extract_ipcc_code_and_short_name(self, var_name, prefix):
        pattern = self._variable_pattern.format(prefix)
        match = re.search(pattern, var_name)
        if match:
            ipcc_code = match.group(1)
            short_name = match.group(2)
            return ipcc_code, short_name
        else:
            return None, None

    def _process(self, data: Dataset) -> Dataset:
        # Drop grid_cell_area variable, use gridcell_area instead
        data = data.drop_vars('grid_cell_area', errors='ignore')

        # Rename variables and add attributes
        name_dict = {}
        for var in data.data_vars:
            attrs = data[var].attrs
            if str(var).startswith(self._emissions_prefix):
                ipcc_code, short_name = self._extract_ipcc_code_and_short_name(var, self._emissions_prefix)
                attrs['IPCC_Code'] = ipcc_code
                attrs['long_name'] = f'{short_name}_Emissions'
                attrs['standard_name'] = self.get_standard_name()
                attrs.pop('source_category', None)  # remove source_category attribute for v2 (same as IPCC code)
                name_dict[var] = short_name
        data = data.rename(name_dict)

        # Round grid to nearest 0.1 degrees
        # - cell coordinates are center-of-cell, so we actually need to round to nearest 0.05
        data = round_latlon(data, self._lat_deci, self._lon_deci)
        return data


class EPAv1(EPA):
    """
    EPA Gridded 2012 Methane Emissions

    https://www.epa.gov/ghgemissions/gridded-2012-methane-emissions

    Maasakkers JD, Jacob DJ, Sulprizio MP, Turner AJ, Weitz M, Wirth T,
    Hight C, DeFigueiredo M, Desai M, Schmeltz R, Hockstad L, Bloom AA,
    Bowman KW, Jeong S, Fischer ML. Gridded National Inventory of U.S.
    Methane Emissions. Environ Sci Technol. 2016 Dec 6;50(23):13123-13133.
    doi: 10.1021/acs.est.6b02878. Epub 2016 Nov 16. PMID: 27934278.
    """
    version: str = 'v1'
    year = 2012

    _emissions_prefix: str = 'emissions'

    def __init__(self, time_step='Annual', inventory_dir: str | None = None) -> None:
        """
        Initialize the EPA inventory.

        Parameters
        ----------
        time_step : str, optional
            The time step of the data, by default 'Annual'.
        inventory_dir : str, optional
            Root of the inventory archive, by default ``$LAIR_INVENTORY_DIR``.
        """
        self.epa_dir = os.path.join(get_data_dir(INVENTORY_DIR_ENV, inventory_dir), 'EPA')
        self.time_step = time_step.lower()
        path = os.path.join(self.epa_dir, self.version, f'GEPA_{self.time_step.capitalize()}.nc')
        super().__init__(path, self.pollutant,
                         src_units=self.src_units, time_step=self.time_step, version=self.version)

    def _process(self, data: Dataset) -> Dataset:
        data = super()._process(data)

        # Format time coordinates
        if self.time_step == 'annual':
            data = data.expand_dims(time=[dt.datetime(self.year, 1, 1)])
        elif self.time_step == 'monthly':
            data = data.assign_coords(time=[dt.datetime(self.year, month, 1)
                                            for month in range(1, 13)])
        elif self.time_step == 'daily':
            data = data.assign_coords(time=[dt.datetime(self.year, 1, 1)
                                            + dt.timedelta(days=i) for i in range(366)])
        else:
            raise ValueError(f'Time step {self.time_step} not supported')
        return data


class EPAv2(EPA):
    """
    EPA U.S. Anthropogenic Methane Greenhouse Gas Inventory

    https://zenodo.org/records/8367082

    McDuffie, Erin, E., Maasakkers, Joannes, D., Sulprizio,
    Melissa, P., Chen, C., Schultz, M., Brunelle, L., Thrush, R.,
    Steller, J., Sherry, C., Jacob, Daniel, J., Jeong, S., Irving,
    B., & Weitz, M. (2023). Gridded EPA U.S. Anthropogenic Methane
    Greenhouse Gas Inventory (gridded GHGI) (v1.0) [Data set]. Zenodo.
    https://doi.org/10.5281/zenodo.8367082
    """
    version: str = 'v2'

    _emissions_prefix: str = 'emi_ch4'
    _express_vars_scalable_past_2018 = [
        'Manure_Management', 'Rice_Cultivation', 'Field_Burning'
    ]

    def __init__(self, express: bool=False, scale_by_month: bool=False,
                 inventory_dir: str | None = None) -> None:
        """
        Initialize the EPA inventory.

        Parameters
        ----------
        express : bool, optional
            Whether to use the express extension, by default False.
        scale_by_month : bool, optional
            Whether to scale emissions by month, by default False.
        inventory_dir : str, optional
            Root of the inventory archive, by default ``$LAIR_INVENTORY_DIR``.
        """
        self.epa_dir = os.path.join(get_data_dir(INVENTORY_DIR_ENV, inventory_dir), 'EPA')
        self.express = express
        self.scale_by_month = scale_by_month

        path = os.path.join(self.epa_dir, self.version, 'express' if express else '')
        super().__init__(path, self.pollutant,
                         src_units=self.src_units, version=self.version)

    def get_monthly_scale_factors(self) -> Dataset:
        """
        Get the monthly scale factors.

        Returns
        -------
        xr.Dataset
            The monthly scale factors.
        """
        files = list(Path(self.epa_dir, self.version, 'monthly_scale_factors'
                          ).glob('*.nc'))
        ds = xr.open_mfdataset(files)
        ds = ds.rename_vars({var: '_'.join(str(var).split('_')[4:])
                             for var in list(ds.data_vars)})
        return ds

    def _scale_by_month(self, data: Dataset) -> Dataset:
        self.time_step = 'monthly'
        sf = self.get_monthly_scale_factors()

        # Round grid coordinates due to floating point errors
        sf = round_latlon(sf, self._lat_deci, self._lon_deci)

        # Filter to variables with strong interannual variability and 
        # expand time dimension by repeating Jan values
        monthly = data[list(sf.data_vars)].reindex(time=sf['time'], method='ffill')
        monthly *= sf  # multiply by scale factors

        if self.express:
            # Only scale _express_vars_scalable_past_2018 past 2018:
            # "For other sources, monthly variability is too year-specific
            # and should not be extrapolated to the express extension dataset
            # for years after 2018"
            sf_2018 = sf[self._express_vars_scalable_past_2018].sel(time='2018')
            express_sf = xr.concat([
                sf_2018.assign_coords(time=np.array(
                    [dt.datetime(year, month, 1) for month in range(1, 13)]
                    ))
                for year in range(2019, int(data.time.dt.year.max()) + 1)
                ], dim='time')

            express_monthly = data[self._express_vars_scalable_past_2018].reindex(time=express_sf['time'], method='ffill')
            express_monthly *= express_sf
            monthly = monthly.combine_first(express_monthly)

            # The other scaled sectors keep their annual rate in every month
            # after 2018 (they would otherwise be NaN, which sums as zero)
            others = [var for var in sf.data_vars
                      if var not in self._express_vars_scalable_past_2018]
            annual_rate = data[others].reindex(time=express_sf['time'], method='ffill')
            monthly = monthly.combine_first(annual_rate)

        return monthly

    def _process(self, data: Dataset) -> Dataset:
        data = super()._process(data)

        if self.scale_by_month:
            data = self._scale_by_month(data)

        return data


class GFEI(Inventory, metaclass=ABCMeta):
    """
    Global Fuel Exploitation Inventory (GFEI)

    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HH4EUM

    This is the Global Fuel Exploitation Inventory (GFEI)
    which provides a 0.1 x 0.1 degree grid of methane emissions
    of fugitive emissions related to oil, gas, and coal activities
    (IPCC Sector 1B1 and 1B2).
    """

    pollutant: str = 'CH4'
    src_units: str = 'Mg km-2 a-1'

    _file_prefix: str

    def __init__(self, inventory_dir: str | None = None) -> None:
        """
        Initialize the GFEI inventory.

        Parameters
        ----------
        inventory_dir : str, optional
            Root of the inventory archive, by default ``$LAIR_INVENTORY_DIR``.
        """
        path = os.path.join(get_data_dir(INVENTORY_DIR_ENV, inventory_dir), 'GFEI', str(self.version))
        super().__init__(path, self.pollutant,
                         src_units=self.src_units, version=self.version)

    def get_files(self) -> list[Path]:
        p = self._file_root()
        return [f for f in p.glob('*.nc')
                if f.stem.split('_')[-1] not in ['All', 'gsd', 'rsd']]

    def _strip_var_names(self, ds: Dataset, suffix: str=''):
        filename = ds.encoding['source']
        var = filename.split(self._file_prefix + '_')[1].split(f'{suffix}.nc')[0]
        return var

    def _preprocess(self, ds: Dataset) -> Dataset:
        # Each file has a single variable named 'emis_ch4'
        # Rename the variable to the filename suffix
        var = self._strip_var_names(ds)
        ds = ds.rename_vars({'emis_ch4': var})
        ds[var].attrs = {
            'long_name':f'{var}_Emissions',
            'standard_name': self.get_standard_name()
        }

        # Add time coordinate
        ds = ds.expand_dims(time=[dt.datetime(int(ds.year), 1, 1)])
        return ds

    def _process(self, data: Dataset) -> Dataset:
        # Drop the 'Total_Fuel_Exploitation' variable (can be calculated)
        data = data.drop_vars(['Total_Fuel_Exploitation'])

        # Even though we aren't quantifying the dims, pint gets mad that
        # the units for lat and lon have spaces in them
        # I feel like this is probably a bug in pint-xarray
        data.lon.attrs['units'] = 'degrees_east'
        data.lat.attrs['units'] = 'degrees_north'

        # Round grid to nearest 0.1 degrees
        # - cell coordinates are center-of-cell, so we actually need to round to nearest 0.05
        data = round_latlon(data, 2, 2)
        return data


class GFEIv1(GFEI):
    """
    Global Fuel Exploitation Inventory v1

    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HH4EUM&version=1.0

    Scarpelli, T. R., Jacob, D. J., Maasakkers, J. D., Sulprizio, M. P.,
    Sheng, J.-X., Rose, K., Romeo, L., Worden, J. R., and Janssens-Maenhout, G.:
    A global gridded (0.1° × 0.1°) inventory of methane emissions from oil, gas,
    and coal exploitation based on national reports to the United Nations Framework
    Convention on Climate Change, Earth Syst. Sci. Data, 12, 563–575,
    https://doi.org/10.5194/essd-12-563-2020, 2020a. 
    """
    version: str = 'v1'
    _file_prefix: str = 'Global_Fuel_Exploitation_Inventory'

    def get_standard_deviations(self, kind: Literal['relative', 'geometric']) -> Dataset:
        """
        Get the standard deviations.

        Parameters
        ----------
        kind : Literal['relative', 'geometric']
            The kind of standard deviation to get ('relative', 'geometric').

        Returns
        -------
        xr.Dataset
            The standard deviations.
        """
        kind_short = {
            'relative':  'rsd',
            'geometric':  'gsd',
        }
        short = kind_short[kind]

        def preprocess(ds: Dataset) -> Dataset:
            # All variables have the same name
            # Rename based on filename
            var = self._strip_var_names(ds, suffix=f'_{short}')
            ds = ds.rename_vars({short: var})
            ds[var].attrs['standard_name'] = f'{kind}_standard_deviation'
            
            # Add time coordinate
            ds = ds.expand_dims(time=[dt.datetime(int(ds.year), 1, 1)])
            return ds

        p = self._file_root()
        files = [f for f in p.glob(f'*{short}.nc')
                 if 'All' not in f.stem]
        sd = xr.open_mfdataset(files, preprocess=preprocess)
        return sd


class GFEIv2(GFEI):
    """
    Global Fuel Exploitation Inventory v2

    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HH4EUM&version=2.0

    Scarpelli, T. R., Jacob, D. J., Grossman, S., Lu, X., Qu, Z.,
    Sulprizio, M. P., Zhang, Y., Reuland, F., Gordon, D., and Worden, J. R.:
    Updated Global Fuel Exploitation Inventory (GFEI) for methane emissions
    from the oil, gas, and coal sectors: evaluation with inversions of
    atmospheric methane observations, Atmos. Chem. Phys., 22, 3235–3249,
    https://doi.org/10.5194/acp-22-3235-2022, 2022.
    """
    version: str = 'v2'
    _file_prefix: str = 'Global_Fuel_Exploitation_Inventory_v2_2019'


class Vulcan(Inventory):
    """
    The Vulcan Project

    https://vulcan.rc.nau.edu/
    v3: https://daac.ornl.gov/NACP/guides/Vulcan_V3_Annual_Emissions.html

    The Vulcan Project quantifies all fossil fuel CO2 emissions for the entire
    United States at high space- and time-resolution with details on economic
    sector, fuel, and combustion process. It was created by the research team
    of Dr. Kevin Robert Gurney at Northern Arizona University.

    Gurney, K.R., J. Liang, R. Patarasuk, Y. Song, J. Huang, and G. Roest. 2019.
    Vulcan: High-Resolution Annual Fossil Fuel CO2 Emissions in USA, 2010-2015,
    Version 3. ORNL DAAC, Oak Ridge, Tennessee, USA.
    https://doi.org/10.3334/ORNLDAAC/1741

    .. note::
        Vulcan distributes emissions as mass of **carbon** (tC, i.e.
        ``Mg C km-2 yr-1``). lair converts them to mass of **CO2** on load
        (multiplying by M(CO2)/M(C) ~= 3.664) so the values are consistent
        with ``pollutant='CO2'`` and with unit conversions to moles. Divide by
        that factor to recover the published tC values. Cells without
        emissions (NaN in the files) are set to 0.
    """
    # hourly data is 1.6 Tb !!!

    version: str = 'v3'
    pollutant: str = 'CO2'
    native_crs = '+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'

    _time_step_dict = {
        'annual': {
            'src_units': 'Mg km-2 a-1',
            'glob_pattern': '*{}.nc4',
            'sep': '_'
        },
        'hourly': {
            'src_units': 'Mg km-2 hr-1',
            'glob_pattern': '*{}.*.*.nc4',
            'sep': '.'
        }
    }
    _uncertainties = {
        'central': 'mn',  # central estimate
        'lower': 'lo',  # lower 95% confidence interval
        'upper': 'hi'  # upper 95% confidence interval
    }

    def __init__(self, time_step: Literal['annual', 'hourly']='annual',
                 region: Literal['US', 'AK']='US',
                 inventory_dir: str | None = None) -> None:
        """
        Initialize the Vulcan inventory.

        Parameters
        ----------
        time_step : Literal['annual', 'hourly'], optional
            The time step of the data, by default 'annual'.
        region : Literal['US', 'AK'], optional
            The region of the data, by default 'US'.
        inventory_dir : str, optional
            Root of the inventory archive, by default ``$LAIR_INVENTORY_DIR``.
        """
        self.vulcan_dir = os.path.join(get_data_dir(INVENTORY_DIR_ENV, inventory_dir), 'vulcan')
        src_units = self._time_step_dict[time_step]['src_units']
        self._glob_pattern = self._time_step_dict[time_step]['glob_pattern']
        self._sep = self._time_step_dict[time_step]['sep']
        self.region = region
        if region == 'AK':
            raise ValueError('Alaska region not supported - issues with 180th meridian')
        path = os.path.join(self.vulcan_dir, self.version, 'data/native', time_step)
        super().__init__(path, self.pollutant,
                         src_units=src_units, time_step=time_step, crs=self.native_crs, version=self.version)
        self._is_clipped = False

    def get_files(self, uncertainty='central') -> list[Path]:
        p = self._file_root()
        uncertainty = self._uncertainties[uncertainty]
        return [f for f in p.glob(self._glob_pattern.format(uncertainty))
                if 'total' not in f.stem
                and self.region in f.stem]

    def get_uncertainties(self, uncertainty: Literal['lower', 'upper']) -> Dataset:
        """
        Get the lower or upper 95% confidence bound of the emissions.

        Parameters
        ----------
        uncertainty : Literal['lower', 'upper']
            Which bound to load.

        Returns
        -------
        xr.Dataset
            The bound for each sector over the full (unclipped) domain, in the
            source units (not pint-quantified).
        """
        if self.time_step != 'annual':
            raise ValueError('Uncertainties are only available for annual data')
        return self._process(self._open(self.get_files(uncertainty)))

    def clip(self,
             bbox: tuple[float, float, float, float] | None = None,
             extent: tuple[float, float, float, float] | None = None,
             geom: Polygon | None = None,
             crs: Any = None,
             inplace: bool = False,
             **kwargs: Any) -> Self:
        """
        Clip the data to the given bounds.

        Input bounds must be in the same CRS as the data.

        .. note::
            The result can be slightly different between supplying a geom and a bbox/extent.
            Clipping with a geom seems to be exclusive of the bounds,
            while clipping with a bbox/extent seems to be inclusive of the bounds.

        Parameters
        ----------
        bbox : tuple[minx, miny, maxx, maxy]
            The bounding box to clip the data to.
        extent : tuple[minx, maxx, miny, maxy]
            The extent to clip the data to.
        geom : shapely.Polygon
            The geometry to clip the data to.
        crs : Any
            The CRS of the input geometries. If not provided, the CRS of the data is used.
        inplace : bool, optional
            Whether to modify the data in place. Default is False (returns a
            new, clipped inventory).
        kwargs : Any
            Additional keyword arguments to pass to the rioxarray clip method.

        Returns
        -------
        Vulcan
            The clipped inventory.
        """
        clipped = super().clip(bbox, extent, geom, crs, inplace=inplace, **kwargs)
        clipped._is_clipped = True
        return clipped

    def reproject(self, resolution: float | tuple[float, float] = 0.01,
                  regrid_method: Regrid_Methods = 'conservative',
                  inplace: bool = False, force: bool = False) -> Self:
        """
        Reproject the data to a lat lon rectilinear grid.
        
        .. tip::
            This method is memory intensive and may require a lot of RAM.
            It is highly recommended to clip the data first.

        Parameters
        ----------
        resolution : float | tuple[x_res, y_res]
            The new resolution in degrees. If a single value is provided, the resolution
            is assumed to be the same in both dimensions.
        regrid_method : str, optional
            The regridding method, by default 'conservative'.
        inplace : bool, optional
            Whether to modify the object in place. Default is False.
        force : bool
            Whether to override the clipping requirement

        Returns
        -------
        Vulcan
            The reprojected inventory (on EPSG:4326 lat/lon).
        """
        if not self._is_clipped and not force:
            raise ValueError('Data must be clipped before reprojecting! Set force=True to override')
        return super().reproject(resolution, regrid_method, inplace=inplace)

    def _preprocess(self, ds: Dataset) -> Dataset:
        # Rename variables
        filename = os.path.basename(ds.encoding['source'])
        sector = filename.split(self._sep)[5]
        ds = ds.rename({'carbon_emissions': sector})
        # Drop unnecessary variables and dims
        ds = ds.drop_vars(['time_bnds', 'crs'])
        return ds

    def _open(self, files: list[Path]) -> Dataset:
        data = xr.open_mfdataset(files, preprocess=self._preprocess,
                                 chunks=None)  # load all data into memory
        data.load()  # FIXME currently, cant have dask chunks and pint units
        # and setting chunks=None is not working 
        return data

    def _process(self, data: Dataset) -> Dataset:
        if self.time_step == 'annual':
            # Set time to first day of year
            data = data.assign_coords(time=[dt.datetime(int(year), 1, 1)
                                            for year in data.time.dt.year])

        # Vulcan marks cells without emissions as NaN (most cells of the point
        # source sectors). Treat them as zero: otherwise every regridded cell
        # touching a NaN becomes NaN and conservative regridding drops most of
        # the airport/cement/cmv/elec emissions.
        data = data.fillna(0)

        # The files are mass of carbon (tC); express them as mass of CO2 to
        # match pollutant='CO2' (see the class note)
        c_to_co2 = float((molecular_weight('CO2') / molecular_weight('C')).magnitude)
        data = data * c_to_co2
        for var in data.data_vars:
            data[var].attrs['comment'] = (f'Converted by lair from tonnes of carbon to '
                                          f'tonnes of CO2 (x {c_to_co2:.4f}).')
        return data


class WetCHARTs(MultiModelInventory):
    """
    WetCHARTs - Wetland Methane Emissions and Uncertainty

    https://daac.ornl.gov/CMS/guides/MonthlyWetland_CH4_WetCHARTs.html

    This dataset provides global monthly wetland methane (CH4) emissions
    estimates at 0.5 by 0.5-degree resolution for the period 2001-2019
    that were derived from an ensemble of multiple terrestrial biosphere
    models, wetland extent scenarios, and CH4:C temperature dependencies
    that encompass the main sources of uncertainty in wetland CH4 emissions.
    There are 18 model configurations. WetCHARTs v1.3.1 is an updated product
    of WetCHARTs v1.0 Extended Ensemble.

    Bloom, A.A., K.W. Bowman, M. Lee, A.J. Turner, R. Schroeder, J.R. Worden,
    R.J. Weidner, K.C. McDonald, and D.J. Jacob. 2021. CMS: Global 0.5-deg
    Wetland Methane Emissions and Uncertainty (WetCHARTs v1.3.1). ORNL DAAC,
    Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/1915
    """
    version: str = 'v1.3.1'
    pollutant = 'CH4'
    src_units: str = 'mg m-2 d-1'
    time_step = 'monthly'

    def __init__(self, model: str | None = None, inventory_dir: str | None = None):
        """
        Initialize the WetCHARTs inventory.

        Parameters
        ----------
        model : str | None, optional
            The model to select, by default None.
            If None, the mean of all models is used.
        inventory_dir : str, optional
            Root of the inventory archive, by default ``$LAIR_INVENTORY_DIR``.
        """
        self.wetcharts_dir = os.path.join(get_data_dir(INVENTORY_DIR_ENV, inventory_dir), 'WetCHARTs')
        path = os.path.join(self.wetcharts_dir, self.version)
        super().__init__(path, self.pollutant,
                         src_units=self.src_units, time_step=self.time_step, version=self.version, model=model)

    def _process(self, data: Dataset) -> Dataset:
        # Drop unnecessary variables and dims
        data = data.drop_vars(['time_bnds', 'crs'])

        # Set time to first day of month
        data = data.assign_coords(time=[dt.datetime(int(year), int(month), 1)
                                       for year, month in zip(data.time.dt.year, data.time.dt.month)])

        # Rename variables
        data = data.rename({'wetland_CH4_emissions': 'wetlands'})
        data.wetlands.attrs['long_name'] = 'Wetland_CH4_Emissions'
        data.wetlands.attrs['standard_name'] = self.get_standard_name()

        return super()._process(data)  # select a single model
