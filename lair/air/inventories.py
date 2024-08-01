from abc import ABCMeta
import datetime as dt
from matplotlib.pyplot import Axes
from molmass import Formula
import numpy as np
import os
from pathlib import Path
import pint
import re
from shapely import Polygon
from typing import Any, Literal
from typing_extensions import Self  # requires python 3.11 to import from typing
import xarray as xr
from xarray import DataArray, Dataset
import xesmf as xe

from lair.config import GROUP_DIR
from lair import units
from lair.utils.clock import TimeRange
from lair.utils.grid import CRS, clip, gridcell_area, wrap_lons

xr.set_options(keep_attrs=True)

# TODO
# Citation & Metadat for each inventory
# Regridding still has some issues, but its pretty close
# I think default units should probably be in kg


#: Inventory directory
INVENTORY_DIR = os.path.join(GROUP_DIR, 'inventories')

DST_UNITS: str = 'umol m-2 s-1'


def molecular_weight(pollutant: str) -> pint.Quantity:
    return Formula(pollutant).mass * units('g/mol')


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
    total = data.to_array().sum('variable')
    total.attrs['long_name'] = 'Total Emissions'

    return total


class Inventory:
    """
    Base class for inventories.
    """

    def __init__(self,
                 data: str | Path | Dataset,
                 pollutant: str,
                 src_units: str | None = None,
                 freq: str = 'annual',
                 crs: str = 'EPSG:4326',
                 version: str | None = None,
                 ) -> None:

        self.pollutant: str = pollutant.upper()
        self.version: str | None = version
        self.freq: str = freq
        self.crs = CRS(crs)

        if isinstance(data, str | Path):
            self.path = str(data)

            # Open dataset
            files = self.get_files()
            data = self._open(files)

            # Set the rioxarray CRS
            data.rio.write_crs(self.crs.to_rasterio(), inplace=True)

            # Apply inventory-specific processing
            data = self._process(data)

            # Standardize units
            # this requires that all variables are emissions, all in the same self.src_units
            data = self._convert_units(self._quantify(data))
        elif  isinstance(data, Dataset):
            self.path = None

            if src_units is None:
                var = list(data.data_vars)[0]
                src_units = data[var].attrs.get('units')
                if src_units is None:
                    raise ValueError('Units must be provided in the data attributes or as an argument')
        else:
            raise ValueError('Data must be a path to a file or an xarray Dataset')

        # Store the data
        self._data: Dataset = data
        self._is_clipped: bool = False  # initially unclipped
        self.src_units: str | pint.registry.Unit = src_units


    def get_standard_name(self) -> str:
        """
        Get the standard name of the inventory.

        Returns
        -------
        str
            The standard name.
        """
        return f'{self.freq.lower()}_{self.pollutant}_emissions'

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

    @property
    def gridcell_area(self) -> DataArray:
        """
        Calculate the grid cell area in km^2.

        Returns
        -------
        xr.DataArray
            The grid cell area.

        Raises
        ------
        ValueError
            If the data is not in EPSG:4326.
        """
        # data must have coords of lat and lon
        if self.data.rio.crs == 'EPSG:4326':
            return gridcell_area(data=self.data)
        else:
            raise ValueError('Data must be in EPSG:4326 to calculate grid cell area')

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
        #  - dimension order of str(pint.Unit) is set here: https://github.com/hgrecco/pint/blob/master/pint/delegates/formatter/full.py#L54
        #  - for mass fluxes, the order will always be [substance | mass] / [area] / [time]
        var = list(self._data.data_vars)[0]  # all variables should be in the same units
        data_units = f'{self._data[var].pint.units: ~C}'  # compact symbols
        _, area_unit, time_unit = data_units.split('/')

        # Multiply by the gridcell area to get mass|substance per time per time step
        absolute = self._data * (self.gridcell_area * units('km**2')).pint.to(area_unit)

        # Get the number of seconds in each time step
        # - I am calculating the exact number of seconds in each time step.
        #   Inventory providers may have used a simpler method of avg secs per time step.
        #   However, its probably close enough to not matter,
        years = absolute.time.dt.year.values
        months = absolute.time.dt.month.values
        days = absolute.time.dt.day.values
        if self.freq == 'annual':
            seconds_per_step = [TimeRange(str(year)).total_seconds
                                for year in years]
        elif self.freq == 'monthly':
            seconds_per_step = [TimeRange(f'{year}-{month:02d}').total_seconds
                                  for year, month in zip(years, months)]
        elif self.freq == 'daily':
            seconds_per_step = [TimeRange(f'{year}-{month:02d}-{day:02d}').total_seconds
                                  for year, month, day in zip(years, months, days)]
        else:
            raise ValueError(f'Frequency {self.freq} not supported')
        seconds_per_step = self._data.assign(sec_per_step=('time', seconds_per_step)).sec_per_step

        # Then multiply by the time in the time step to get mass|substance per gridcell
        absolute = absolute * (seconds_per_step * units('s')).pint.to(time_unit)

        absolute.attrs = {'long_name': 'Absolute Emissions',
                          'standard_name': f'{self.freq.lower()}_emissions_per_gridcell'}
        return absolute.pint.dequantify()

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
    def data(self) -> Dataset:
        """
        The inventory data.

        .. note::
            `pint` units have been dequantified and stored in variable attributes.

        Returns
        -------
        xr.DataArray | xr.Dataset
            The inventory data.
        """
        return self._data.pint.dequantify()

    @data.setter
    def data(self, data: Dataset) -> None:
        self._data = self._quantify(data, use_attrs=True)

    def quantify(self) -> Dataset:
        """
        Quantify the data using `pint` units for each variable.

        Returns
        -------
        xr.Dataset
            The quantified data.
        """
        return self._data

    def convert_units(self, dst_units: Any) -> Self:
        """
        Convert the units of the data to the desired output units.

        Parameters
        ----------
        dst_units : Any
            The destination units.

        Returns
        -------
        xr.Dataset
            The data with the units converted.
        """
        self._data = self._convert_units(self._data, dst_units)

        return self

    def clip(self,
             bbox: tuple[float, float, float, float] | None = None,
             extent: tuple[float, float, float, float] | None = None,
             geom: Polygon | None = None,
             crs: Any = None,
             **kwargs: Any) -> Self:
        """
        Clip the data to the given bounds. Modifies the data in place.

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
        kwargs : Any
            Additional keyword arguments to pass to the rioxarray clip method.

        Returns
        -------
        Inventory
            The clipped inventory.
        """
        crs = crs or self.crs.to_rasterio()
        self._data = clip(self._data, bbox=bbox, extent=extent, geom=geom, crs=crs, **kwargs)
        self._is_clipped = True

        return self

    def regrid(self, out_grid: Dataset) -> Self:
        """
        Regrid the data to a new grid.

        .. warning::
            `xarray.Dataset.cf.add_bounds` is known to have issues, including near the 180th meridian.
            Care should be taken when using this method, especially with global datasets.

        Parameters
        ----------
        out_grid : xr.DataArray
            The new grid to resample to.

        Returns
        -------
        Inventory
            The regridded inventory.
        """
        # Use cf-xarray to calculate the bounds of the grid cells
        data = self.data.cf.add_bounds(['lat', 'lon'])

        # Regrid the data using a conservative regridder
        regridder = xe.Regridder(ds_in=data, ds_out=out_grid, method='conservative_normed')
        self.data = regridder(data, keep_attrs=True)  # setting on self.data will update self._data

        return self

    def resample(self, resolution: float | tuple[float, float]) -> Self:
        """
        Resample the data to a new resolution.

        Parameters
        ----------
        resolution : float | tuple[x_res, y_res]
            The new resolution in degrees. If a single value is provided, the resolution
            is assumed to be the same in both dimensions.

        Returns
        -------
        Inventory
            The resampled inventory.
        """
        if isinstance(resolution, float):
            resolution = (resolution, resolution)

        # Calculate the new grid
        bounds = self.data.cf.add_bounds(['lat', 'lon'])
        xmin = bounds.lon_bounds.min()
        xmax = bounds.lon_bounds.max()
        ymin = bounds.lat_bounds.min()
        ymax = bounds.lat_bounds.max()
        dx = resolution[0]
        dy = resolution[1]
        if len(self.data.lon.dims) == 2:
            out_grid = xe.util.grid_2d(xmin, xmax, dx,
                                       ymin, ymax, dy)
        else:
            out_grid = xr.Dataset({
                "lat": (["lat"], np.arange(ymin, ymax+dy, dy), {"units": "degrees_north"}),
                "lon": (["lon"], np.arange(xmin, xmax+dx, dx), {"units": "degrees_east"}),
                })
        return self.regrid(out_grid)

    def reproject(self, resolution: float | tuple[float, float]) -> Self:
        """
        Reproject the data to a lat lon rectilinear grid.

        Parameters
        ----------
        resolution : float | tuple[x_res, y_res]
            The new resolution in degrees. If a single value is provided, the resolution
            is assumed to be the same in both dimensions.

        Returns
        -------
        Inventory
            The reprojected inventory.
        """
        assert self.crs.epsg != 4326, 'Data is already in lat lon'
        return self.resample(resolution)

    def integrate(self) -> DataArray:
        """
        Integrate the data over the spatial dimensions
        to get the total emissions per time period.

        Returns
        -------
        xr.DataArray
            The integrated data.
        """
        return sum_sectors(self.absolute_emissions.sum(['lat', 'lon']))

    def plot(self, ax: Axes, sector: str | None = None, **kwargs) -> Axes:
        """
        Plot the inventory data.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on.
        sector : str, optional
            The sector to plot, by default None. If None, the total emissions are plotted.
        kwargs : dict
            Additional keyword arguments to pass to `xarray`'s plot method.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        """
        if sector is not None:
            data = self.data[sector]
        else:
            data = self.total_emissions

        data.plot(ax=ax, transform=self.crs.to_cartopy(), **kwargs)

        return ax

    def _open(self, files: list[Path]) -> Dataset:
        # Open the dataset, preprocessing if necessary
        return xr.open_mfdataset(files, preprocess=getattr(self, '_preprocess', None))

    def _process(self, data) -> Dataset:
        # In the base case, we just return the data
        # This method can be overridden in the subclasses to process the data
        # Resulting ds hould have coords of time, lat, lon and all variables should be emissions
        return data

    def _quantify(self, data: Dataset, use_attrs: bool = False) -> Dataset:
        # Quantify the data using `pint` units
        src_units = None if use_attrs else self.src_units
        for var in data.data_vars:
            data[var] = data[var].pint.quantify(src_units)
        return data

    def _convert_units(self, data: Dataset, dst_units: Any = None) -> Dataset:
        dst_units = dst_units or DST_UNITS

        # Calculate molecular weight of pollutant
        mw = molecular_weight(self.pollutant)

        # Use custom pint context to convert mass <--> substance
        with units.context('mass_flux', mw=mw):
            for var in data.data_vars:
                data[var] = data[var].pint.to(dst_units)
                data[var].attrs['units'] = DST_UNITS  # FIXME
        return data


class MultiModelInventory(Inventory, metaclass=ABCMeta):
    """
    Base class for inventories that are multi-model.
    """
    multimodel_data: Dataset

    def __init__(self,
                 data: str | Path | Dataset,
                 pollutant: str,
                 src_units: str | None = None,
                 freq: str = 'annual',
                 crs: str = 'EPSG:4326',
                 version: str | None = None,
                 model: str|None = None) -> None:
        self.model = model or 'mean'
        super().__init__(data, pollutant, src_units, freq, crs, version)

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
    """
    edgar_dir = os.path.join(INVENTORY_DIR, 'EDGAR')
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
        return [f for f in Path(self.path).rglob('*.nc')
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
        print(description, name)
        return name


class EDGARv7(EDGAR):
    version: str = 'v7'

    def __init__(self, pollutant: str):
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

        return data


class EDGARv8(EDGAR):
    version: str = 'v8'

    def __init__(self, pollutant: str, freq: Literal['annual', 'monthly']='annual'):
        path = os.path.join(self.edgar_dir, self.version,
                            '' if freq == 'annual' else freq, pollutant)
        super().__init__(path, pollutant,
                         src_units=self.src_units, freq=freq, version=self.version)

    def _preprocess(self, ds: Dataset) -> Dataset:
        # Rename var and add attributes
        old_var = 'fluxes'
        var = ds[old_var].attrs['long_name'].replace(' ', '_').replace(',', '').replace('-', '_')
        ds = ds.rename({old_var: var})
        attrs = ds[var].attrs
        attrs['long_name'] = f'{var}_Emissions'
        attrs['standard_name'] = self.get_standard_name()

        if self.freq == 'annual':
            # Add time coordinate to annual data
            ds = ds.expand_dims(time=[dt.datetime(int(attrs['year']), 1, 1)])
        return ds

    def _process(self, data: Dataset) -> Dataset:
        # Fuel_exploitation is the sum of Fuel_exploitation_COAL, Fuel_exploitation_GAS, and Fuel_exploitation_OIL
        data = data.drop_vars(['Fuel_exploitation'], errors='ignore')
        return data


class EPA(Inventory, metaclass=ABCMeta):
    """
    EPA Greenhouse Gas Inventory
    """
    epa_dir: str = os.path.join(INVENTORY_DIR, 'EPA')
    pollutant: str = 'CH4'
    src_units: str = 'molec cm-2 s-1'

    _emissions_prefix: str
    _variable_pattern: str = r'{}(?:_Supp)?_([1-9][A-Z]?[1-9]?[a-z]*)_([A-Za-z_]*)'

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
            if var.startswith(self._emissions_prefix):
                ipcc_code, short_name = self._extract_ipcc_code_and_short_name(var, self._emissions_prefix)
                attrs['IPCC_Code'] = ipcc_code
                attrs['long_name'] = f'{short_name}_Emissions'
                attrs['standard_name'] = self.get_standard_name()
                attrs.pop('source_category', None)  # remove source_category attribute for v2 (same as IPCC code)
                name_dict[var] = short_name
        data = data.rename(name_dict)

        # Round grid to nearest 0.1 degrees
        data = data.assign_coords(lon=data.lon.round(2),
                                  lat=data.lat.round(2))
        return data


class EPAv1(EPA):
    """
    EPA Greenhouse Gas Inventory v1
    """
    version: str = 'v1'
    year = 2012

    _emissions_prefix: str = 'emissions'

    def __init__(self, freq='Annual'):
        self.freq = freq.lower()
        path = os.path.join(self.epa_dir, self.version, f'GEPA_{self.freq.capitalize()}.nc')
        super().__init__(path, self.pollutant,
                         src_units=self.src_units, freq=self.freq, version=self.version)

    def _process(self, data: Dataset) -> Dataset:
        data = super()._process(data)

        # Format time coordinates
        if self.freq == 'annual':
            data = data.expand_dims(time=[dt.datetime(self.year, 1, 1)])
        elif self.freq == 'monthly':
            data = data.assign_coords(time=[dt.datetime(self.year, month, 1)
                                            for month in range(1, 13)])
        elif self.freq == 'daily':
            data = data.assign_coords(time=[dt.datetime(self.year, 1, 1)
                                            + dt.timedelta(days=i) for i in range(366)])
        else:
            raise ValueError(f'Frequency {self.freq} not supported')
        return data


class EPAv2(EPA):
    version: str = 'v2'

    _emissions_prefix: str = 'emi_ch4'
    _express_vars_scalable_past_2018 = [
        'Manure_Management', 'Rice_Cultivation', 'Field_Burning'
    ]

    def __init__(self, express: bool=False, scale_by_month: bool=False):
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
        ds = ds.rename_vars({var: '_'.join(var.split('_')[4:])
                             for var in list(ds.data_vars)})
        return ds

    def _scale_by_month(self, data: Dataset) -> Dataset:
        self.freq = 'monthly'
        sf = self.get_monthly_scale_factors()

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

        return monthly

    def _process(self, data: Dataset) -> Dataset:
        data = super()._process(data)

        if self.scale_by_month:
            data = self._scale_by_month(data)

        return data


class GFEI(Inventory, metaclass=ABCMeta):
    """
    Global Fuel Exploitation Inventory 
    """

    pollutant: str = 'CH4'
    src_units: str = 'Mg km-2 a-1'

    _file_prefix: str

    def __init__(self):
        path = os.path.join(INVENTORY_DIR, 'GFEI', self.version)
        super().__init__(path, self.pollutant,
                         src_units=self.src_units, version=self.version)

    def get_files(self) -> list[Path]:
        p = Path(self.path)
        return [f for f in p.glob('*.nc')
                if not f.stem.split('_')[-1] in ['All', 'gsd', 'rsd']]

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
        return data.drop_vars(['Total_Fuel_Exploitation'])


class GFEIv1(GFEI):
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

        p = Path(self.path)
        files = [f for f in p.glob(f'*{short}.nc')
                 if 'All' not in f.stem]
        sd = xr.open_mfdataset(files, preprocess=preprocess)
        return sd


class GFEIv2(GFEI):
    version: str = 'v2'
    _file_prefix: str = 'Global_Fuel_Exploitation_Inventory_v2_2019'


class Vulcan(Inventory):
    # hourly data is 1.6 Tb !!!
    vulcan_dir = os.path.join(INVENTORY_DIR, 'vulcan')

    version: str = 'v3'
    pollutant: str = 'CO2'
    crs = '+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'

    _freq_dict = {
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

    def __init__(self, freq: Literal['annual', 'hourly']='annual', region: Literal['US', 'AK']='US'):

        src_units = self._freq_dict[freq]['src_units']
        self._glob_pattern = self._freq_dict[freq]['glob_pattern']
        self._sep = self._freq_dict[freq]['sep']
        self.region = region
        if region == 'AK':
            raise ValueError('Alaska region not supported - issues with 180th meridian')
        path = os.path.join(self.vulcan_dir, self.version, 'data/native', freq)
        super().__init__(path, self.pollutant,
                         src_units=src_units, freq=freq, crs=self.crs, version=self.version)

    def get_files(self, uncertainty='central') -> list[Path]:
        p = Path(self.path)
        uncertainty = self._uncertainties[uncertainty]
        return [f for f in p.glob(self._glob_pattern.format(uncertainty))
                if 'total' not in f.stem
                and self.region in f.stem]

    def get_uncertainties(self, uncertainty: Literal['lower', 'upper']) -> Dataset:
        assert self.freq == 'annual', 'Uncertainties are only available for annual data'
        files = self.get_files(self._uncertainties[uncertainty])
        return xr.open_mfdataset(files)

    def reproject(self, resolution: float | tuple[float, float] = 0.01,
                  force: bool = False) -> Self:
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
        force : bool
            Whether to override the clipping requirement

        Returns
        -------
        Inventory
            The reprojected inventory.
        """
        if not self._is_clipped and not force:
            raise ValueError('Data must be clipped before reprojecting! Set force=True to override')
        return super().reproject(resolution)

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
        if self.freq == 'annual':
            # Set time to first day of year
            data = data.assign_coords(time=[dt.datetime(int(year), 1, 1)
                                            for year in data.time.dt.year])

        return data


class WetCHARTs(MultiModelInventory):
    wetcharts_dir = os.path.join(INVENTORY_DIR, 'WetCHARTs')
    version: str = 'v1.3.1'
    pollutant = 'CH4'
    src_units: str = 'mg m-2 d-1'
    freq = 'monthly'

    def __init__(self, model: str | None = None):
        path = os.path.join(self.wetcharts_dir, self.version)
        super().__init__(path, self.pollutant,
                         src_units=self.src_units, freq=self.freq, version=self.version, model=model)

    def _process(self, data: Dataset) -> Dataset:
        # Drop unnecessary variables and dims
        data = data.drop_vars(['time_bnds', 'crs']).drop_dims('nv')

        # Rename variables
        data = data.rename({'wetland_CH4_emissions': 'wetlands'})
        data.wetlands.attrs['long_name'] = 'Wetland_CH4_Emissions'
        data.wetlands.attrs['standard_name'] = self.get_standard_name()

        return super()._process(data)  # select a single model
