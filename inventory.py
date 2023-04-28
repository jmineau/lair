#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:00:42 2023

@author: James Mineau (James.Mineau@utah.edu)

Module for working with Emission Inventories
"""

import os
import rioxarray as rxr


INVENTORY_DIR = ('/uufs/chpc.utah.edu/common/home/u6036966'
                 '/wkspace/data/EmissionInventories')

# TODO - use this instead, need to add to it
LAIR_inventory_dir = ('/uufs/chpc.utah.edu/common/home/lin-group9/inventories')


# Unit Conversions
avogadro = 6.02214076e23  # molec / mol
molec_to_mol = 1 / avogadro
mol_to_umol = 1e6

metric_tonnes_to_g = 1e6
kg_to_g = 1000
Mg_to_g = 1e6
Gg_to_g = 1e9
g_CH4_to_mol_C = 1 / 16.04
g_C_to_mol_C = 1 / 12.01

cm2_to_m2 = 1e-4
km2_to_m2 = 1e6

a_to_s = 3.1536e7


def convertunits(ds, converter, units='umol m-2 s-1'):
    # TODO create FluxUnitConverter class
    ds *= converter
    ds.attrs['units'] = units

    return None


class Inventory:
    def __init__(self, ID, specie, sector, file):
        self.ID = ID
        self.specie = specie
        self.sector = sector
        self.file = file
        self.path = self.get_path(file)
        # TODO accept units kw

    def __repr__(self):
        return (f'Inventory(ID="{self.ID}", specie="{self.specie}", '
                f'sector="{self.sector}")')

    def __copy__(self):
        return

    def get_path(self, file):
        return os.path.join(INVENTORY_DIR, self.ID, file)

    def process(self, path):
        ds = rxr.open_rasterio(path)
        ds.rio.write_crs(4326, inplace=True)
        return ds

    def clip(self, geom, box=False, crs=None):
        from collections.abc import Sequence

        if box:
            clipped = self.inventory.rio.clip_box(*geom, crs=crs)
        else:
            if not isinstance(geom, Sequence):
                geom = [geom]

            clipped = self.inventory.rio.clip(geom, crs=crs)

        return self.__class__._from_clip(clipped, self)

    def apply_weights(self):
        import numpy as np

        weights = np.cos(np.deg2rad(self.inventory[self.inventory.rio.y_dim]))
        weighted = self.inventory.weighted(weights)
        return weighted

    def get_cell_area(self, inventory):
        from helper.grid import area_DataArray

        area = area_DataArray(inventory)

        return area

    def integrate(self):
        flux_per_cell = self.inventory * self.area
        integrated_flux = flux_per_cell.sum(dim=[self.inventory.rio.y_dim,
                                                 self.inventory.rio.x_dim])

        return integrated_flux

    def add2map(self, ax, **kwargs):
        # TODO
        #   use grid.CRS_Converter to convert to cartopy crs
        import cartopy.crs as ccrs
        self.inventory.plot(ax=ax, transform=ccrs.PlateCarree(),
                            **kwargs)
        ax.set(title=None,
               xlabel=None,
               ylabel=None)

    @classmethod
    def _from_clip(cls, clip, old_obj):
        obj = cls.__new__(cls)
        obj.__dict__.update(old_obj.__dict__)

        obj.inventory = clip
        obj.weighted = obj.apply_weights()
        obj.area = obj.get_cell_area(clip)
        obj.integrated = obj.integrate()

        return obj


class CSL(Inventory):
    def __init(self, specie):
        ID = 'CSL'
        file = ''
        super().__init__(ID, specie, file)


class EDGAR(Inventory):
    version_file = {'7.0': 'EDGARv7_{specie}_total_2021.nc'}

    def __init__(self, specie, version='7.0'):
        ID = 'EDGAR'
        self.version = version
        sector = 'total'
        file = EDGAR.version_file[version].format(specie=specie)

        super().__init__(ID, specie, sector, file)

        self.inventory = self.post_process(self.process(self.path))
        self.weighted = self.apply_weights()
        self.area = self.get_cell_area(self.inventory)
        self.integrated = self.integrate()

    def post_process(self, ds):
        from helper.grid import wrap_lons

        # Edgar coordinates are the cell center of each grid-cell
        # Convert lon from 0~360 -> -180~180
        ds.coords['x'] = wrap_lons(ds.coords['x'])
        ds = ds.sortby(ds.x)

        # kg m-2 s-1 -> umol m-2 s-1
        converter = kg_to_g * g_CH4_to_mol_C * mol_to_umol
        convertunits(ds, converter)

        return ds


class EPA(Inventory):
    def __init__(self, sector='total'):
        ID = 'EPA'
        specie = 'CH4'
        file = 'GEPA_Annual.nc'

        super().__init__(ID, specie, sector, file)

        self.inventory = self.post_process(self.process(self.path), sector)
        self.weighted = self.apply_weights()
        self.area = self.get_cell_area(self.inventory)
        self.integrated = self.integrate()

    def post_process(self, ds, sector):
        ds = self.subset(ds, sector)

        # molecules CH4 per cm2 per s -> umol m-2 s-1
        converter = molec_to_mol * mol_to_umol / cm2_to_m2
        convertunits(ds, converter)

        return ds

    def subset(self, ds, sector):
        if sector == 'all':
            return ds

        elif sector == 'total':
            # Save attributes
            attrs = ds.attrs
            # Important attributes are saved within variables attrs
            #    Pull first variable to get attrs
            var_attrs = ds.emissions_1A_Combustion_Mobile.attrs
            var_attrs.pop('standard_name', None)  # remove standard_name

            total = ds.to_array().sum('variable')  # sum across variables
            total = total.assign_attrs(attrs).assign_attrs(var_attrs)
            return total

        else:
            # Dict of short name: full name for each sector
            sectors = {'_'.join(var.split('_', 2)[2:]).lower(): var for var in
                       list(ds.variables) if var.startswith('emissions')}

            return ds[sectors[sector.lower()]]


class GFEI(Inventory):
    def __init__(self, version=2, sector='Total_Fuel_Exploitation',
                 subsector=None):
        ID = 'GFEI'
        specie = 'CH4'
        file = GFEI.get_file(version, sector, subsector)

        super().__init__(ID, specie, sector, file)

        self.inventory = self.post_process(self.process(self.path))
        self.weighted = self.apply_weights()
        self.area = self.get_cell_area(self.inventory)
        self.integrated = self.integrate()

    def post_process(self, ds):
        # Mg a-1 km-2 -> umol m-2 s-1
        converter = Mg_to_g * g_CH4_to_mol_C * mol_to_umol / km2_to_m2 / a_to_s
        convertunits(ds, converter)

        return ds

    @classmethod
    def get_file(cls, version, sector, subsector):
        file = ('Global_Fuel_Exploitation_Inventory_'
                f'{"v2_2019_" if version == 2 else ""}{sector}'
                f'{"_"+subsector if subsector else ""}.nc')

        return file


class Hestia(Inventory):
    def __init__(self):
        pass


class SUMRF(Inventory):
    def __init__(self):
        pass


class Vulcan(Inventory):
    def __init__(self, year='2015', version=3, sector='total',
                 uncertainty='mn'):
        ID = 'Vulcan'
        specie = 'CO2'
        file = Vulcan.get_file(version, sector, uncertainty)

        super().__init__(ID, specie, sector, file)

        self.inventory = self.post_process(self.process(self.path, year))
        self.weighted = self.apply_weights()
        self.area = self.get_cell_area(self.inventory)
        self.integrated = self.integrate()

    def process(self, path, year):
        from rasterio.enums import Resampling
        import xarray as xr
        # Overwrite Inventory.process since Vulcan has non latlon projection
        ds = xr.open_dataset(path)
        da = ds.sel(time=year).carbon_emissions.drop_vars(['lat', 'lon'])

        # Gridcells are 1000m x1000m (1 km2), meaning the value of the cell
        #   is the total flux through the cell per time
        # Reprojecting to a resolution of 0.002 deg results in gridcells that
        #   are about 27x smaller than before... keep cells in units of per
        #   area so that changing cell resolution does not impact the results

        proj_str = '+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'
        da.rio.write_crs(proj_str, inplace=True)
        da_proj = da.rio.reproject(4326, resolution=0.002,
                                   resampling=Resampling.bilinear)

        return da_proj

    def post_process(self, ds):
        # metric tonnes carbon per km2 per year -> umol/m2/s
        converter = (metric_tonnes_to_g * g_C_to_mol_C * mol_to_umol
                     / km2_to_m2 / a_to_s)
        convertunits(ds, converter)
        return ds

    @classmethod
    def get_file(cls, version, sector, uncertainty):
        file = (f'v{version}/Vulcan_v{version}_US_annual_1km_'
                f'{sector}_{uncertainty}.nc4')

        # Thredds server
        # this one works for rxr
        'https://thredds.daac.ornl.gov/thredds/fileServer/ornldaac/1741/Vulcan_v3_US_annual_1km_total_mn.nc4'

        # this one for xr
        'https://thredds.daac.ornl.gov/thredds/dodsC/ornldaac/1741/Vulcan_v3_US_annual_1km_total_mn.nc4'

        return file
